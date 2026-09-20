from pathlib import Path

from dashboard.data import build_dashboard_bundle, save_dashboard_bundle


def main() -> None:
    output = Path("data/processed/dashboard_bundle.pkl")
    bundle = build_dashboard_bundle()
    save_dashboard_bundle(bundle, output)
    print(f"dashboard snapshot: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
