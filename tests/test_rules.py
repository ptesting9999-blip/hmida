import pytest

from leadgen.verify import rules


@pytest.mark.parametrize("email", ["jane@acme.com", "jane.doe+leads@acme.co.uk", "j_o'neil@sub.acme.io"])
def test_valid_syntax(email):
    assert rules.check_syntax(email) is None


@pytest.mark.parametrize(
    "email",
    ["jane@@acme.com", "jane", ".jane@acme.com", "ja..ne@acme.com", "jane@acme", "jane@-acme.com", "a" * 65 + "@acme.com"],
)
def test_invalid_syntax(email):
    assert rules.check_syntax(email) is not None


@pytest.mark.parametrize(
    "email",
    ["info@acme.com", "Support@acme.com", "hello2@acme.com", "customer-care@acme.com", "sales.team@acme.com", "no-reply@acme.com", "info+x@acme.com"],
)
def test_role_accounts(email):
    assert rules.is_role_account(email)


@pytest.mark.parametrize("email", ["jane@acme.com", "jane.sales@acme.com", "mike.info@acme.com", "ceo.jane@acme.com"])
def test_personal_accounts(email):
    assert not rules.is_role_account(email)


def test_disposable_matches_subdomains():
    disposable = rules.load_disposable_domains()
    assert rules.is_disposable("x@mailinator.com", disposable)
    assert rules.is_disposable("x@eu.mailinator.com", disposable)
    assert not rules.is_disposable("x@acme.com", disposable)


def test_free_providers():
    assert rules.is_free_provider("jane@gmail.com")
    assert rules.is_free_provider("outlook.com")
    assert not rules.is_free_provider("jane@acme.com")
