"""
AlphaFlow Conformational Exploration Service
GPU-accelerated conformational ensemble generation on Azure Container Apps.

Wraps AlphaFlow/ESMFlow inference in a FastAPI service with async job tracking,
Azure Blob result storage, and quanta-mcp compatible Redis status updates.

Port 8025 - Internal service
"""

import os
import json
import logging
import tempfile
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx
import redis.asyncio as aioredis
from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "8025"))
API_KEY = os.environ.get("API_KEY", "alphaflow-api-key-2024")
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))

# Model weights
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/opt/alphaflow/params/esmflow_md_base_202402.pt")

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "alphaflow-results")

# Redis
REDIS_URL = os.environ.get("AZURE_REDIS_URL", os.environ.get("REDIS_URL", "redis://localhost:6379"))

# ---------------------------------------------------------------------------
# Globals (initialized on startup)
# ---------------------------------------------------------------------------
redis_client: Optional[aioredis.Redis] = None
blob_service_client: Optional[BlobServiceClient] = None
gpu_semaphore: Optional[asyncio.Semaphore] = None
model = None  # Loaded once on startup

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AlphaFlow Conformational Exploration Service",
    description="AI-accelerated conformational ensemble generation using ESMFlow",
    version="1.0.0",
    root_path="/alphaflow",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class DynamicsRequest(BaseModel):
    pdb_data: Optional[str] = Field(None, description="PDB file content as string")
    pdb_id: Optional[str] = Field(None, description="PDB ID to fetch from RCSB")
    sequence: Optional[str] = Field(None, description="Amino acid sequence (if no PDB)")
    n_frames: int = Field(50, ge=5, le=500, description="Number of conformations to generate")
    steps: int = Field(10, ge=5, le=50, description="Number of diffusion steps")

class JobResponse(BaseModel):
    job_id: str
    status: str
    estimated_runtime_minutes: int
    poll_url: str

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def validate_api_key(api_key: str = Header(None, alias="API-Key")):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model():
    """Load ESMFlow model into GPU memory. Called once on startup."""
    import torch
    from alphaflow.model.wrapper import ESMFoldWrapper
    from alphaflow.config import model_config

    config = model_config("initial_training", train=True, low_prec=True)
    config.data.common.use_templates = False
    config.data.common.max_recycling_iters = 0

    logger.info(f"Loading ESMFlow weights from {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    wrapper = ESMFoldWrapper(**ckpt["hyper_parameters"], training=False)
    wrapper.model.load_state_dict(ckpt["params"], strict=False)
    wrapper = wrapper.cuda()
    wrapper.eval()
    logger.info("ESMFlow model loaded and ready")
    return wrapper, config

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global redis_client, blob_service_client, gpu_semaphore, model

    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    # Redis
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis not available: {e}. Job tracking will be in-memory only.")
        redis_client = None

    # Azure Blob Storage
    if AZURE_STORAGE_CONNECTION_STRING:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(
                AZURE_STORAGE_CONNECTION_STRING
            )
            try:
                blob_service_client.create_container(AZURE_STORAGE_CONTAINER)
            except Exception:
                pass  # Already exists
            logger.info(f"Connected to Azure Blob Storage (container: {AZURE_STORAGE_CONTAINER})")
        except Exception as e:
            logger.warning(f"Azure Blob Storage not available: {e}")
            blob_service_client = None
    else:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set. Results stored locally only.")

    # Load model into GPU
    try:
        model = load_model()
        logger.info("GPU model ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

    logger.info(f"AlphaFlow v1.0.0 started on port {PORT}")
    logger.info(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.aclose()

# ---------------------------------------------------------------------------
# Redis Job Tracking (compatible with quanta-mcp polling)
# ---------------------------------------------------------------------------
async def update_job_status(job_id: str, status: str, progress: Dict[str, Any]):
    """Update job status in Redis using quanta-mcp compatible format."""
    if not redis_client:
        return
    try:
        key = f"quanta-mcp:job:{job_id}"
        data = {
            "job_id": job_id,
            "status": status,
            "progress": json.dumps(progress),
            "last_updated": datetime.utcnow().isoformat(),
        }
        await redis_client.hset(key, mapping=data)
        await redis_client.expire(key, 86400)  # 24 hours
        logger.info(f"Job {job_id}: {progress.get('percentage', 0)}% - {progress.get('message', '')}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


async def complete_job(job_id: str, result: Dict[str, Any]):
    """Mark job as completed in Redis (3-key format for quanta-mcp compatibility)."""
    if not redis_client:
        return
    try:
        now = datetime.utcnow().isoformat()
        progress = {"percentage": 100, "message": "Conformational ensemble generated", "step": "completed"}

        # 1. Unified hash (what quanta-mcp reads)
        unified_key = f"quanta-mcp:job:{job_id}"
        await redis_client.hset(unified_key, mapping={
            "job_id": job_id,
            "status": "completed",
            "completed_at": now,
            "result": json.dumps(result),
            "progress": json.dumps(progress),
            "last_updated": now,
        })
        await redis_client.expire(unified_key, 604800)  # 7 days

        # 2. Legacy result key
        result_key = f"quanta-mcp:job_result:{job_id}"
        await redis_client.set(result_key, json.dumps(result), ex=604800)

        # 3. Legacy cache key
        cache_key = f"quanta-mcp:cache:jobs:{job_id}"
        cache_data = {
            "job_id": job_id,
            "status": "completed",
            "completed_at": now,
            "result": result,
            "progress": progress,
            "last_updated": now,
        }
        await redis_client.set(cache_key, json.dumps(cache_data), ex=604800)

        logger.info(f"Job {job_id} completed")
    except Exception as e:
        logger.error(f"Failed to complete job {job_id}: {e}")


async def fail_job(job_id: str, error: str):
    """Mark job as failed in Redis."""
    if not redis_client:
        return
    try:
        await update_job_status(job_id, "failed", {
            "percentage": 0,
            "message": f"Generation failed: {error}",
            "step": "failed",
            "error": error,
        })
    except Exception as e:
        logger.error(f"Failed to mark job {job_id} as failed: {e}")

# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
async def upload_results_to_blob(pdb_content: str, job_id: str, metadata: Dict) -> Optional[str]:
    """Upload ensemble PDB and metadata to Azure Blob Storage."""
    if not blob_service_client:
        logger.warning("Blob storage not available, skipping upload")
        return None

    loop = asyncio.get_event_loop()
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
    prefix = f"alphaflow/{job_id}"

    # Upload multi-model PDB
    pdb_blob = f"{prefix}/ensemble.pdb"
    await loop.run_in_executor(
        None,
        lambda: container_client.upload_blob(pdb_blob, pdb_content, overwrite=True),
    )

    # Upload metadata
    meta_blob = f"{prefix}/metadata.json"
    await loop.run_in_executor(
        None,
        lambda: container_client.upload_blob(
            meta_blob, json.dumps(metadata, indent=2), overwrite=True
        ),
    )

    logger.info(f"Uploaded results for {job_id} to Azure Blob")
    return f"{prefix}/ensemble.pdb"

# ---------------------------------------------------------------------------
# PDB Fetching & Sequence Extraction
# ---------------------------------------------------------------------------
async def fetch_pdb_from_rcsb(pdb_id: str) -> str:
    """Fetch PDB file content from RCSB."""
    pdb_id = pdb_id.strip().upper()
    # Try PDB format first, then mmCIF
    async with httpx.AsyncClient(timeout=30) as client:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text
        # Try mmCIF fallback
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text
        raise HTTPException(status_code=404, detail=f"PDB {pdb_id} not found on RCSB")


def extract_sequence_from_pdb(pdb_content: str) -> tuple[str, str]:
    """Extract amino acid sequence and name from PDB content.

    Returns (sequence, name) tuple.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1
    import io

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_content))

    # Use the first chain
    for model_obj in structure:
        for chain in model_obj:
            residues = []
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in protein_letters_3to1:
                    residues.append(protein_letters_3to1[resname])
            if residues:
                name = f"{structure.header.get('idcode', 'unknown')}_{chain.id}"
                return "".join(residues), name

    raise ValueError("No protein chain found in PDB content")


def estimate_runtime_minutes(sequence_length: int, n_frames: int) -> int:
    """Estimate runtime based on sequence length and frame count.

    Empirical: ~0.5-2s per frame for 300-residue protein on A100.
    Scales roughly quadratically with sequence length (attention).
    """
    # Base: 1 second per frame for a 300-residue protein
    base_seconds_per_frame = 1.0
    length_factor = (sequence_length / 300) ** 2
    total_seconds = n_frames * base_seconds_per_frame * length_factor
    # Add overhead: model loading (already loaded), PDB fetch, RMSF computation
    total_seconds += 30
    return max(1, round(total_seconds / 60))

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    sequence: str,
    name: str,
    n_frames: int,
    steps: int,
    update_callback=None,
) -> tuple[str, Dict[str, Any]]:
    """Run ESMFlow inference synchronously (called via run_in_executor).

    Returns (multi_model_pdb, metadata) tuple.
    """
    import torch
    import numpy as np
    import time
    from alphaflow.data.data_modules import collate_fn
    from alphaflow.data.inference import seq_to_tensor
    from openfold.data.data_transforms import make_atom14_masks
    import alphaflow.utils.protein as protein

    wrapper, config = model

    # Build batch from sequence (same as CSVDataset.__getitem__)
    batch = {
        "name": name,
        "seqres": sequence,
        "aatype": seq_to_tensor(sequence),
        "residue_index": torch.arange(len(sequence)),
        "pseudo_beta_mask": torch.ones(len(sequence)),
        "seq_mask": torch.ones(len(sequence)),
    }
    make_atom14_masks(batch)

    # Diffusion schedule
    tmax = 1.0
    schedule = np.linspace(tmax, 0, steps + 1)

    # Generate conformations
    results = []
    runtimes = []
    collated = collate_fn([batch])
    from alphaflow.utils.tensor_utils import tensor_tree_map
    collated = tensor_tree_map(lambda x: x.cuda(), collated)

    for i in range(n_frames):
        start = time.time()
        with torch.no_grad():
            prots = wrapper.inference(
                collated,
                as_protein=True,
                noisy_first=False,
                no_diffusion=False,
                schedule=schedule,
                self_cond=False,
            )
        runtimes.append(time.time() - start)
        results.append(prots[-1])

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Generated frame {i + 1}/{n_frames} ({runtimes[-1]:.1f}s)")

    # Build multi-model PDB
    pdb_content = protein.prots_to_pdb(results)

    # Compute per-residue RMSF from ensemble
    rmsf = compute_rmsf(results)

    # Compute PCA summary
    pca_summary = compute_pca(results)

    metadata = {
        "name": name,
        "sequence_length": len(sequence),
        "n_frames": n_frames,
        "steps": steps,
        "total_runtime_seconds": round(sum(runtimes), 1),
        "avg_seconds_per_frame": round(sum(runtimes) / len(runtimes), 2),
        "rmsf_per_residue": rmsf,
        "pca_summary": pca_summary,
        "generated_at": datetime.utcnow().isoformat(),
    }

    return pdb_content, metadata


def compute_rmsf(prots) -> List[float]:
    """Compute per-residue RMSF (root mean square fluctuation) from ensemble.

    Uses C-alpha positions across all conformations.
    """
    import numpy as np

    # Extract C-alpha positions (index 1 in atom14 representation)
    ca_positions = []
    for prot in prots:
        # prot.atom_positions shape: [num_res, num_atom_type, 3]
        ca = prot.atom_positions[:, 1, :]  # C-alpha
        ca_positions.append(ca)

    ca_positions = np.array(ca_positions)  # [n_frames, n_residues, 3]

    # Mean position per residue
    mean_pos = ca_positions.mean(axis=0)  # [n_residues, 3]

    # RMSF = sqrt(mean(||pos - mean_pos||^2))
    deviations = ca_positions - mean_pos[np.newaxis, :, :]
    sq_dev = np.sum(deviations**2, axis=2)  # [n_frames, n_residues]
    rmsf = np.sqrt(sq_dev.mean(axis=0))  # [n_residues]

    return [round(float(v), 3) for v in rmsf]


def compute_pca(prots, n_components: int = 3) -> Dict[str, Any]:
    """Compute PCA of conformational variation from C-alpha positions.

    Returns explained variance ratios and per-frame projections.
    """
    import numpy as np

    ca_positions = []
    for prot in prots:
        ca = prot.atom_positions[:, 1, :]  # C-alpha [n_res, 3]
        ca_positions.append(ca.flatten())

    ca_matrix = np.array(ca_positions)  # [n_frames, n_res*3]

    # Center
    mean = ca_matrix.mean(axis=0)
    centered = ca_matrix - mean

    # SVD-based PCA (avoids covariance matrix for large proteins)
    n_comp = min(n_components, centered.shape[0] - 1, centered.shape[1])
    if n_comp < 1:
        return {"explained_variance_ratio": [], "projections": []}

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (S**2) / (S**2).sum()

    projections = centered @ Vt[:n_comp].T  # [n_frames, n_comp]

    return {
        "explained_variance_ratio": [round(float(v), 4) for v in explained_var[:n_comp]],
        "projections": [[round(float(v), 3) for v in row] for row in projections],
    }

# ---------------------------------------------------------------------------
# Background Task
# ---------------------------------------------------------------------------
async def _run_dynamics_job(
    job_id: str,
    sequence: str,
    name: str,
    n_frames: int,
    steps: int,
):
    """Background task: run inference and store results."""
    try:
        async with gpu_semaphore:
            await update_job_status(job_id, "running", {
                "percentage": 10,
                "message": f"Generating {n_frames} conformations for {name} ({len(sequence)} residues)",
                "step": "inference",
            })

            # Run inference in executor (blocking GPU work)
            loop = asyncio.get_event_loop()
            pdb_content, metadata = await loop.run_in_executor(
                None,
                run_inference,
                sequence,
                name,
                n_frames,
                steps,
            )

            await update_job_status(job_id, "running", {
                "percentage": 90,
                "message": "Uploading results",
                "step": "upload",
            })

            # Upload to blob storage
            blob_path = await upload_results_to_blob(pdb_content, job_id, metadata)

            # Build result
            result = {
                "job_id": job_id,
                "name": name,
                "sequence_length": metadata["sequence_length"],
                "n_frames": metadata["n_frames"],
                "total_runtime_seconds": metadata["total_runtime_seconds"],
                "pdb_ensemble": pdb_content,
                "rmsf_per_residue": metadata["rmsf_per_residue"],
                "pca_summary": metadata["pca_summary"],
                "generated_at": metadata["generated_at"],
            }
            if blob_path:
                result["blob_path"] = blob_path

            await complete_job(job_id, result)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await fail_job(job_id, str(e))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "alphaflow",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": gpu_available,
        "model_loaded": model is not None,
        "redis_connected": redis_ok,
        "blob_storage_connected": blob_service_client is not None,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "port": PORT,
    }


@app.get("/about")
async def about():
    return {
        "service": "AlphaFlow Conformational Exploration",
        "version": "1.0.0",
        "description": "AI-accelerated conformational ensemble generation using ESMFlow",
        "capabilities": [
            "Generate conformational ensembles from protein sequence",
            "Accept PDB ID, PDB content, or raw amino acid sequence",
            "Per-residue RMSF (flexibility) analysis",
            "PCA of conformational variation",
            "Multi-model PDB output for NGL trajectory animation",
        ],
        "model": "ESMFlow (MD-distilled, base)",
        "gpu_required": True,
    }


@app.post("/generate", dependencies=[Depends(validate_api_key)], response_model=JobResponse)
async def generate_dynamics(request: DynamicsRequest):
    """Submit conformational ensemble generation job.

    Accepts PDB data, PDB ID, or raw sequence. Returns job_id immediately —
    inference runs asynchronously in the background.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. GPU may not be available.")

    # Resolve input to sequence + name
    sequence = None
    name = None

    if request.pdb_data:
        # Extract sequence from PDB content
        await update_job_status("resolving", "running", {
            "percentage": 0, "message": "Extracting sequence from PDB", "step": "resolve",
        })
        try:
            sequence, name = extract_sequence_from_pdb(request.pdb_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDB: {e}")

    elif request.pdb_id:
        # Fetch from RCSB and extract sequence
        try:
            pdb_content = await fetch_pdb_from_rcsb(request.pdb_id)
            sequence, name = extract_sequence_from_pdb(pdb_content)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch/parse PDB {request.pdb_id}: {e}")

    elif request.sequence:
        # Raw sequence
        sequence = request.sequence.strip().upper()
        name = f"seq_{uuid.uuid4().hex[:8]}"
        # Basic validation
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(sequence) - valid_aa
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid amino acid characters: {invalid}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: pdb_data, pdb_id, or sequence",
        )

    if len(sequence) > 2000:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence too long ({len(sequence)} residues). Maximum is 2000.",
        )

    # Create job
    job_id = f"af_{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    est_minutes = estimate_runtime_minutes(len(sequence), request.n_frames)

    # Initialize in Redis
    await update_job_status(job_id, "queued", {
        "percentage": 0,
        "message": f"Queued: {name} ({len(sequence)} residues, {request.n_frames} frames)",
        "step": "queued",
    })
    if redis_client:
        try:
            key = f"quanta-mcp:job:{job_id}"
            await redis_client.hset(key, mapping={
                "submitted_at": datetime.utcnow().isoformat(),
                "estimated_runtime_minutes": str(est_minutes),
            })
        except Exception:
            pass

    # Launch background task
    asyncio.create_task(_run_dynamics_job(
        job_id=job_id,
        sequence=sequence,
        name=name,
        n_frames=request.n_frames,
        steps=request.steps,
    ))

    return JobResponse(
        job_id=job_id,
        status="queued",
        estimated_runtime_minutes=est_minutes,
        poll_url=f"/alphaflow/status/{job_id}",
    )


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status from Redis."""
    if not redis_client:
        return {"job_id": job_id, "status": "unknown", "message": "Redis not available"}

    try:
        key = f"quanta-mcp:job:{job_id}"
        data = await redis_client.hgetall(key)

        if not data:
            return {"job_id": job_id, "status": "not_found", "progress": 0}

        progress = {}
        if "progress" in data:
            try:
                progress = json.loads(data["progress"])
            except (json.JSONDecodeError, TypeError):
                pass

        status = data.get("status", "unknown")
        response = {
            "job_id": job_id,
            "status": status,
            "progress": progress.get("percentage", 0),
            "message": progress.get("message", ""),
            "step": progress.get("step", ""),
            "last_updated": data.get("last_updated"),
        }

        # Estimated remaining time for running jobs
        if status in ("queued", "running"):
            submitted_at = data.get("submitted_at")
            est_total = data.get("estimated_runtime_minutes")
            if submitted_at and est_total:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(submitted_at)).total_seconds() / 60
                    remaining = max(0, round(int(est_total) - elapsed))
                    response["estimated_remaining_minutes"] = remaining
                    response["estimated_total_minutes"] = int(est_total)
                    response["elapsed_minutes"] = round(elapsed)
                except (ValueError, TypeError):
                    pass

        return response
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {"job_id": job_id, "status": "error", "message": str(e)}


@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get conformational ensemble results from Redis."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Try unified hash first
        key = f"quanta-mcp:job:{job_id}"
        data = await redis_client.hgetall(key)

        if data and data.get("status") == "completed":
            result = {}
            if "result" in data:
                try:
                    result = json.loads(data["result"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return {"job_id": job_id, "status": "completed", "result": result}

        # Try legacy result key
        result_key = f"quanta-mcp:job_result:{job_id}"
        result_str = await redis_client.get(result_key)
        if result_str:
            return {"job_id": job_id, "status": "completed", "result": json.loads(result_str)}

        if data:
            status = data.get("status", "unknown")
            progress = {}
            if "progress" in data:
                try:
                    progress = json.loads(data["progress"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return {"job_id": job_id, "status": status, "progress": progress}

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward-compatible alias
@app.get("/jobs/{job_id}/status")
async def get_job_status_alias(job_id: str):
    return await get_job_status(job_id)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
