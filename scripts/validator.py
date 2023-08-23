"""Validate stuff."""
from .validator_best_is_best import main as best_is_best

VALIDATORS = {"best_is_best": best_is_best}


def main() -> None:
    """Run all validators."""
    for name, validator in VALIDATORS.items():
        print("VALIDATE", name)
        try:
            validator()
        except AssertionError as e:
            print("VALIDATION FAILED")
            print(e)


if __name__ == '__main__':
    main()
