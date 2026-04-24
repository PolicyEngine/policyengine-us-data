import importlib
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python .github/scripts/check_python_import.py <module>[:attribute]"
        )

    target = sys.argv[1]
    module_name, _, attribute_name = target.partition(":")
    module = importlib.import_module(module_name)
    if attribute_name:
        getattr(module, attribute_name)
    print("OK")


if __name__ == "__main__":
    main()
