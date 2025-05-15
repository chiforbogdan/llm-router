1. Edit gpu_cluster.yaml and configure the desired resources
2. python -m venv llm-router && source llm-router/bin/activate
3. pip install -r requirements.txt
4. ray up cluster.yaml
5. ray dashboard cluster.yaml (if GPU VRAM is not visible do steps 6,7).
6. ray down cluster.yaml
7. ray up cluster.yaml
8. While doing ray <yaml> dashboard, RAY_ADDRESS=http://localhost:8265 serve deploy serve_latest.yaml.
Before do this: cd src; ray rsync_up ../gpu_cluster.yaml . '/home/ray' -v
9. To re-deploy: RAY_ADDRESS=http://localhost:8265 serve shutdown and RAY_ADDRESS=http://localhost:8265 serve deploy serve_latest.yaml.
10. To execute queires: cd src; python request_client.py

