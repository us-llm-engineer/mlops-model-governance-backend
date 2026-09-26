"""Entry point: python -m mlops.svc [--config FILE] [--host H] [--port P] [--check]."""

import argparse
import sys

from mlops.kernel import ValidationFailed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mlops.svc", description="MLOps control plane service")
    parser.add_argument("--config", metavar="FILE", default=None, help="JSON settings file")
    # 0.0.0.0 so a containerized deployment (e.g. behind a Helm-managed Service/Ingress) is
    # reachable from outside the container without remembering to override this every time.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true", help="build the app, print ok, exit without serving")
    args = parser.parse_args(argv)

    from mlops.svc.app import create_app, serve
    from mlops.svc.settings import load_settings

    try:
        settings = load_settings(args.config)
        app = create_app(settings)
    except ValidationFailed as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2

    if args.check:
        print("ok")
        return 0

    # `app` above is discarded here: --check only proved settings parse and a
    # minimal app builds. The real run uses serve(), which wires in the
    # extensions/workers/OpsRepo this service actually ships with (see
    # svc.app.build_default_wiring) -- create_app(settings) alone never does.
    serve(settings, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
