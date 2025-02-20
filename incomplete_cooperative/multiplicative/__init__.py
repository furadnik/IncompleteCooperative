"""Module for multiplicative approximation."""
from .max_xos_approximation import compute_max_xos_approximation

APPROXIMATORS = {
    "max_xos": compute_max_xos_approximation
}
