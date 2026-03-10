from cosmic_memory.graph.identity import (
    deterministic_identity_key_id,
    normalize_email,
    normalize_name_variant,
)
from cosmic_memory.graph.ontology import IdentityKeyType


def test_normalize_email_is_case_insensitive_and_gmail_aware():
    assert normalize_email("User.Name+tag@Gmail.com") == "username@gmail.com"


def test_deterministic_identity_key_id_is_stable_for_same_email():
    normalized = normalize_email("User@UALR.edu")
    first = deterministic_identity_key_id(IdentityKeyType.EMAIL, normalized)
    second = deterministic_identity_key_id(IdentityKeyType.EMAIL, normalized)

    assert first == second


def test_normalize_name_variant_strips_honorifics():
    assert normalize_name_variant("Dr. Nitin Agarwal") == "nitin agarwal"
