from severewx.cli import build_tornado_concern_spc_window_products as spc_window_products_cli


def test_parse_spc_day1_valid_window_from_outlook_text() -> None:
    text = """
    Day 1 Convective Outlook
    NWS Storm Prediction Center Norman OK

    Valid 181300Z - 191200Z
    """

    start, end = spc_window_products_cli.parse_spc_day1_valid_window(text, init_date="2026-05-18")

    assert start == "2026-05-18T13:00Z"
    assert end == "2026-05-19T12:00Z"


def test_parse_spc_day1_valid_window_handles_month_rollover() -> None:
    text = "Valid 312000Z - 011200Z"

    start, end = spc_window_products_cli.parse_spc_day1_valid_window(text, init_date="2026-05-31")

    assert start == "2026-05-31T20:00Z"
    assert end == "2026-06-01T12:00Z"
