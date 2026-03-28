# MCP Tool Registration for generate_dynamics

Add the following to `quanta-mcp/mcp/tools.py` in NovoQuantNexus:

## 1. Tool Definition (add to MCP_TOOLS dict)

```python
    "generate_dynamics": {
        "name": "generate_dynamics",
        "title": "Generate Conformational Dynamics",
        "description": "Generate AI-accelerated conformational ensemble from a protein structure. Shows how a protein moves — loops swaying, domains shifting, binding pockets opening/closing. Returns a job_id — use get_job_status to poll for results every 60s. Results include multi-model PDB for trajectory animation, per-residue RMSF (flexibility), and PCA of conformational variation.",
        "tier": ToolTier.ENTERPRISE,
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "pdb_id": {
                    "type": "string",
                    "description": "PDB ID of the protein (e.g., '1M17')"
                },
                "pdb_data": {
                    "type": "string",
                    "description": "PDB file content as string (alternative to pdb_id)"
                },
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence (alternative to pdb_id/pdb_data)"
                },
                "n_frames": {
                    "type": "integer",
                    "description": "Number of conformations to generate (5-500, default 50). Higher = more detail but longer runtime.",
                    "default": 50,
                    "minimum": 5,
                    "maximum": 500
                }
            },
            "required": []
        }
    },
```

## 2. Credit Cost (add to TOOL_CREDITS dict)

```python
    "generate_dynamics": 25,  # GPU-tier: A100 for 1-5 minutes
```

## 3. Service Routing (add to __init__ service_api_keys)

```python
    "alphaflow": os.getenv("ALPHAFLOW_API_KEY") or internal_api_key,
```

## 4. Service URL (add to service_urls config/environment)

```
ALPHAFLOW_URL=https://alphaflow.internal.ashymoss-d55ab909.eastus.azurecontainerapps.io
```

## 5. Job ID Prefix Routing (add to _execute_get_job_status)

```python
    elif job_id.startswith("af_"):
        service = "alphaflow"
```

And in `service_endpoints`:

```python
    "alphaflow": ("alphaflow", f"/status/{job_id}"),
```

## 6. Execution Method

```python
    async def _execute_generate_dynamics(self, args: Dict[str, Any], context: Dict[str, Any] = None) -> ToolResult:
        """Generate conformational ensemble via AlphaFlow/ESMFlow."""
        pdb_id = args.get("pdb_id")
        pdb_data = args.get("pdb_data")
        sequence = args.get("sequence")
        n_frames = args.get("n_frames", 50)

        if not pdb_id and not pdb_data and not sequence:
            return ToolResult(success=False, error="Provide one of: pdb_id, pdb_data, or sequence")

        payload = {"n_frames": n_frames}
        if pdb_id:
            payload["pdb_id"] = pdb_id.upper()
        elif pdb_data:
            payload["pdb_data"] = pdb_data
        elif sequence:
            payload["sequence"] = sequence

        try:
            response = await self._call_service(
                "alphaflow",
                "/generate",
                payload,
                timeout=120.0,
            )

            if response.status_code not in (200, 202):
                return ToolResult(success=False, error=f"AlphaFlow service error: {response.status_code}")

            data = response.json()
            job_id = data.get("job_id", "")
            estimated_minutes = data.get("estimated_runtime_minutes", 3)

            # Persist to async_jobs
            try:
                await self._execute_save_funnel_context({
                    "job_id": job_id,
                    "service": "alphaflow",
                    "context": {
                        "funnel_step": 9,
                        "tool": "generate_dynamics",
                        "pdb_id": pdb_id,
                        "n_frames": n_frames,
                    }
                }, context=context)
            except Exception:
                pass

            return ToolResult(
                success=True,
                data={
                    "job_id": job_id,
                    "status": "submitted",
                    "pdb_id": pdb_id,
                    "n_frames": n_frames,
                    "estimated_minutes": estimated_minutes,
                    "message": (
                        f"Conformational ensemble generation submitted"
                        f"{' for ' + pdb_id if pdb_id else ''}. "
                        f"Generating {n_frames} conformations. "
                        f"Estimated runtime: ~{estimated_minutes} minutes. "
                        f"Use get_job_status with job_id '{job_id}' — poll every 60s until completed. "
                        f"Results will include animated trajectory, per-residue RMSF, and PCA of motions."
                    ),
                    "tool_suggestions": [
                        self._tool_suggestion("get_job_status",
                            f"Check dynamics job {job_id} (wait ~{estimated_minutes} min, then poll every 60s)"),
                    ]
                },
                usage={"queries": 1, "tool": "generate_dynamics"}
            )

        except Exception as e:
            logger.exception(f"Error in generate_dynamics: {e}")
            return ToolResult(success=False, error=f"Dynamics generation failed: {str(e)}")
```

## 7. Structure Viewer App (novomcp-apps)

Add a "Generate Dynamics" button to the Structure Viewer MCP app in `novomcp-apps/src/structure-viewer.tsx`:

```typescript
// When user clicks "Generate Dynamics":
const result = await callServerTool({
  name: "generate_dynamics",
  arguments: {
    pdb_data: currentPdbContent,
    n_frames: 50,
  },
});
// Result contains job_id — show pending state, poll with get_job_status
```
