"""Check for what percentage of games is the divergence supermodular."""
from random import randint

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.generators import convex_generator
from incomplete_cooperative.norms import l1_norm
from incomplete_cooperative.supermodularity_check import check_supermodularity

divergence = l1_norm


def main() -> None:
    """Repeatedly check the divergence for supermodularity."""
    while True:
        g = convex_generator(randint(5, 10))
        supermodularity = check_supermodularity(g)
        print(g.get_values())
        if supermodularity:
            print(supermodularity)
            a, b, i = supermodularity
            ic = Coalition.from_players([i])
            print(g.get_value(a), g.get_value(a | ic), g.get_value(b), g.get_value(b | ic))
            print(g.get_value(a | ic) - g.get_value(a), g.get_value(b | ic) - g.get_value(b))
        else:
            print("ok")


if __name__ == '__main__':
    main()
