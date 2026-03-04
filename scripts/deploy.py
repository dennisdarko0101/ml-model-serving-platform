#!/usr/bin/env python3
"""CLI deployment script for the ML Model Serving Platform.

Usage:
    python scripts/deploy.py --target local --action deploy
    python scripts/deploy.py --target cloud-run --action deploy --service-name ml-serving
    python scripts/deploy.py --target cloud-run --action status --service-name ml-serving
    python scripts/deploy.py --target cloud-run --action rollback --service-name ml-serving
"""

from __future__ import annotations

import argparse
import sys

from ml_serving.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy ML Model Serving Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["local", "cloud-run"],
        required=True,
        help="Deployment target",
    )
    parser.add_argument(
        "--action",
        choices=["deploy", "update", "rollback", "status", "delete"],
        required=True,
        help="Deployment action",
    )
    parser.add_argument("--service-name", default="ml-serving", help="Cloud Run service name")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--memory", default="512Mi", help="Memory allocation")
    parser.add_argument("--cpu", default="1", help="CPU allocation")
    parser.add_argument("--min-instances", type=int, default=0, help="Minimum instances")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum instances")
    parser.add_argument("--image", default="", help="Docker image URI")
    parser.add_argument("--tag", default="latest", help="Image tag")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.target == "local":
        _handle_local(args)
    elif args.target == "cloud-run":
        _handle_cloud_run(args)


def _handle_local(args: argparse.Namespace) -> None:
    from ml_serving.deployment.docker_deploy import DockerDeployer

    deployer = DockerDeployer()

    if args.action == "deploy":
        print("=== Local Docker Deployment ===")
        print(f"  Port: {args.port}")

        if not args.yes:
            confirm = input("\nProceed? [y/N] ")
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        tag = args.image or "ml-serving:latest"
        print(f"\nBuilding image: {tag}")
        deployer.build(tag=tag)

        print(f"Starting container on port {args.port}...")
        info = deployer.run(tag, name=args.service_name, port=args.port)
        print(f"Container started: {info.name} ({info.container_id})")
        print(f"  URL: http://localhost:{args.port}")
        print(f"  Health: http://localhost:{args.port}/health")

    elif args.action == "delete":
        print(f"Stopping container: {args.service_name}")
        deployer.stop(args.service_name)
        print("Done.")

    elif args.action == "status":
        url = f"http://localhost:{args.port}/health"
        healthy = deployer.health_check(url)
        print(f"Service {'healthy' if healthy else 'unhealthy'}: {url}")

    else:
        print(f"Action '{args.action}' not supported for local target")
        sys.exit(1)


def _handle_cloud_run(args: argparse.Namespace) -> None:
    from ml_serving.deployment.cloud_run import CloudRunDeployer

    settings = get_settings()
    project_id = settings.GCP_PROJECT
    if not project_id:
        print("Error: GCP_PROJECT not set. Set it in .env or environment.")
        sys.exit(1)

    deployer = CloudRunDeployer(project_id=project_id, region=args.region)

    if args.action == "deploy":
        # Show cost estimate
        cost = deployer.estimate_cost(
            memory_mb=_parse_memory_mb(args.memory),
            cpu=float(args.cpu),
        )
        print("=== Cloud Run Deployment ===")
        print(f"  Project:  {project_id}")
        print(f"  Region:   {args.region}")
        print(f"  Service:  {args.service_name}")
        print(f"  Memory:   {args.memory}")
        print(f"  CPU:      {args.cpu}")
        print(f"  Min/Max:  {args.min_instances}/{args.max_instances}")
        print(f"\n--- Estimated Monthly Cost ---")
        print(f"  Requests: ${cost['request_cost_usd']:.2f}")
        print(f"  CPU:      ${cost['cpu_cost_usd']:.2f}")
        print(f"  Memory:   ${cost['memory_cost_usd']:.2f}")
        print(f"  Total:    ${cost['total_estimated_usd']:.2f}/month")

        if not args.yes:
            confirm = input("\nProceed with deployment? [y/N] ")
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        image_uri = args.image
        if not image_uri:
            print("\nBuilding image...")
            image_uri = deployer.build_image("ml-serving", tag=args.tag)

        print(f"Deploying to Cloud Run: {args.service_name}")
        info = deployer.deploy(
            image_uri,
            args.service_name,
            memory=args.memory,
            cpu=args.cpu,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
        )
        print(f"\nDeployment complete!")
        print(f"  URL:      {info.url}")
        print(f"  Revision: {info.revision}")
        print(f"  Status:   {info.status}")

    elif args.action == "update":
        if not args.image:
            print("Error: --image required for update")
            sys.exit(1)
        info = deployer.update(args.service_name, args.image)
        print(f"Updated: {info.service_name} -> {info.image_uri}")

    elif args.action == "status":
        status = deployer.get_status(args.service_name)
        print(f"Service: {status.service_name}")
        print(f"  URL:      {status.url}")
        print(f"  Revision: {status.latest_revision}")
        print(f"  Status:   {status.status}")
        print(f"  Ready:    {status.ready}")

    elif args.action == "rollback":
        print(f"Rollback not available via CLI — use the API or Python SDK.")
        print(f"The RollbackManager requires checkpoint history from a running session.")

    elif args.action == "delete":
        if not args.yes:
            confirm = input(f"Delete service '{args.service_name}'? [y/N] ")
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)
        deployer.delete(args.service_name)
        print(f"Deleted: {args.service_name}")


def _parse_memory_mb(memory: str) -> int:
    """Parse memory string like '512Mi' or '1Gi' to MB."""
    mem = memory.strip()
    if mem.endswith("Gi"):
        return int(float(mem[:-2]) * 1024)
    elif mem.endswith("Mi"):
        return int(mem[:-2])
    return 512


if __name__ == "__main__":
    main()
