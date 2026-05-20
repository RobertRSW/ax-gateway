import json
from unittest.mock import MagicMock

from typer.testing import CliRunner

from ax_cli.commands.spaces import _bound_agent_allows_space, _find_space, _space_items, _space_label
from ax_cli.main import app

runner = CliRunner()


# ---------- _space_items ----------


def test_space_items_from_list():
    assert _space_items([{"id": "1"}, {"id": "2"}]) == [{"id": "1"}, {"id": "2"}]


def test_space_items_from_list_filters_non_dicts():
    assert _space_items([{"id": "1"}, "bad", 42]) == [{"id": "1"}]


def test_space_items_from_dict_spaces_key():
    assert _space_items({"spaces": [{"id": "s1"}]}) == [{"id": "s1"}]


def test_space_items_from_dict_items_key():
    assert _space_items({"items": [{"id": "i1"}]}) == [{"id": "i1"}]


def test_space_items_from_dict_results_key():
    assert _space_items({"results": [{"id": "r1"}]}) == [{"id": "r1"}]


def test_space_items_returns_empty_for_non_dict_non_list():
    assert _space_items("string") == []
    assert _space_items(42) == []
    assert _space_items(None) == []


def test_space_items_dict_no_matching_key():
    assert _space_items({"other": [{"id": "x"}]}) == []


# ---------- _space_label ----------


def test_space_label_uses_slug():
    assert _space_label({"slug": "my-slug", "name": "My Name"}, "fb") == "my-slug"


def test_space_label_uses_name_when_no_slug():
    assert _space_label({"name": "My Name"}, "fb") == "My Name"


def test_space_label_uses_space_name_when_no_slug_or_name():
    assert _space_label({"space_name": "SN"}, "fb") == "SN"


def test_space_label_uses_fallback():
    assert _space_label({}, "fb") == "fb"


# ---------- _find_space ----------


def test_find_space_returns_matching_space():
    client = MagicMock()
    client.list_spaces.return_value = [
        {"id": "aaa", "name": "A"},
        {"id": "bbb", "name": "B"},
    ]
    assert _find_space(client, "bbb") == {"id": "bbb", "name": "B"}


def test_find_space_matches_on_space_id_key():
    client = MagicMock()
    client.list_spaces.return_value = [{"space_id": "ccc", "name": "C"}]
    assert _find_space(client, "ccc") == {"space_id": "ccc", "name": "C"}


def test_find_space_returns_none_when_not_found():
    client = MagicMock()
    client.list_spaces.return_value = [{"id": "aaa"}]
    assert _find_space(client, "zzz") is None


def test_find_space_returns_none_on_exception():
    client = MagicMock()
    client.list_spaces.side_effect = RuntimeError("boom")
    assert _find_space(client, "aaa") is None


# ---------- _bound_agent_allows_space ----------


def test_bound_agent_allows_space_returns_true_when_space_in_list():
    client = MagicMock()
    client.whoami.return_value = {
        "bound_agent": {
            "agent_name": "bot",
            "allowed_spaces": [{"space_id": "s1"}],
        }
    }
    allowed, name = _bound_agent_allows_space(client, "s1")
    assert allowed is True
    assert name == "bot"


def test_bound_agent_allows_space_returns_false_when_not_in_list():
    client = MagicMock()
    client.whoami.return_value = {
        "bound_agent": {
            "agent_name": "bot",
            "allowed_spaces": [{"space_id": "s1"}],
        }
    }
    allowed, name = _bound_agent_allows_space(client, "s2")
    assert allowed is False
    assert name == "bot"


def test_bound_agent_allows_none_none_when_whoami_fails():
    client = MagicMock()
    client.whoami.side_effect = RuntimeError("fail")
    allowed, name = _bound_agent_allows_space(client, "s1")
    assert allowed is None
    assert name is None


def test_bound_agent_allows_none_none_when_no_bound_agent():
    client = MagicMock()
    client.whoami.return_value = {}
    allowed, name = _bound_agent_allows_space(client, "s1")
    assert allowed is None
    assert name is None


def test_bound_agent_allows_none_name_when_no_allowed_spaces_list():
    client = MagicMock()
    client.whoami.return_value = {"bound_agent": {"agent_name": "bot"}}
    allowed, name = _bound_agent_allows_space(client, "s1")
    assert allowed is None
    assert name == "bot"


# ---------- list_spaces command ----------


def test_list_spaces_json_via_gateway(monkeypatch):
    monkeypatch.setattr(
        "ax_cli.commands.spaces.resolve_gateway_config",
        lambda: {"some": "cfg"},
    )
    monkeypatch.setattr(
        "ax_cli.commands.messages._gateway_local_call",
        lambda gateway_cfg, method: [{"id": "s1", "name": "Space1", "slug": "space-1", "member_count": 5}],
    )
    result = runner.invoke(app, ["spaces", "list", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["id"] == "s1"


def test_list_spaces_text_via_client(monkeypatch):
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_gateway_config", lambda: {})

    client = MagicMock()
    client.list_spaces.return_value = [{"id": "s1", "name": "Space1", "slug": "space-1", "member_count": 2}]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "list"])
    assert result.exit_code == 0, result.output
    assert "Space1" in result.output


def test_list_spaces_text_table_columns(monkeypatch):
    """Regression for #49/#50: default table must include the Slug column
    (sole disambiguator for same-name spaces) and must NOT include a
    Visibility column the API doesn't populate."""
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_gateway_config", lambda: {})

    client = MagicMock()
    client.list_spaces.return_value = [
        {"id": "s1", "name": "Space1", "slug": "space-1", "member_count": 2},
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "list"])
    assert result.exit_code == 0, result.output
    assert "Slug" in result.output
    assert "space-1" in result.output
    assert "Visibility" not in result.output


def test_list_spaces_text_disambiguates_same_name(monkeypatch):
    """Regression for #49: when two spaces share a name, the slug column
    is the only visible disambiguator in the default table — without it,
    an operator hitting the #47/#48 ambiguity error can't see which slug
    is which."""
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_gateway_config", lambda: {})

    client = MagicMock()
    client.list_spaces.return_value = [
        {"id": "11111111-1111-1111-1111-111111111111", "name": "Demo Team", "slug": "demo-team-1", "member_count": 1},
        {"id": "22222222-2222-2222-2222-222222222222", "name": "Demo Team", "slug": "demo-team-2", "member_count": 1},
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "list"])
    assert result.exit_code == 0, result.output
    assert "demo-team-1" in result.output
    assert "demo-team-2" in result.output


def test_list_spaces_unwraps_dict_response(monkeypatch):
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_gateway_config", lambda: {})

    client = MagicMock()
    client.list_spaces.return_value = {
        "spaces": [{"id": "s1", "name": "SpaceWrapped", "visibility": "public", "member_count": 1}]
    }
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "list", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["name"] == "SpaceWrapped"


def test_list_spaces_unwraps_items_key(monkeypatch):
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_gateway_config", lambda: {})

    client = MagicMock()
    client.list_spaces.return_value = {"items": [{"id": "s1", "name": "ItemSpace"}]}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "list", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["name"] == "ItemSpace"


# ---------- create command ----------


def test_create_space_json(monkeypatch):
    client = MagicMock()
    client.create_space.return_value = {"space": {"id": "new-id", "name": "NewSpace", "visibility": "private"}}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "create", "NewSpace", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["name"] == "NewSpace"


def test_create_space_text(monkeypatch):
    client = MagicMock()
    client.create_space.return_value = {"space": {"id": "new-id-1234", "name": "MySpace", "visibility": "public"}}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "create", "MySpace", "-d", "desc", "-v", "public"])
    assert result.exit_code == 0, result.output
    assert "Created" in result.output
    assert "MySpace" in result.output


def test_create_space_flat_result(monkeypatch):
    client = MagicMock()
    client.create_space.return_value = {"id": "flat-id", "name": "Flat", "visibility": "private"}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "create", "Flat", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["name"] == "Flat"


# ---------- get_space command ----------


def test_get_space_json(monkeypatch):
    client = MagicMock()
    client.get_space.return_value = {"id": "s1", "name": "SpaceGet", "visibility": "private"}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "get", "s1", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["name"] == "SpaceGet"


def test_get_space_text(monkeypatch):
    client = MagicMock()
    client.get_space.return_value = {"id": "s1", "name": "SpaceGet", "visibility": "private"}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "get", "s1"])
    assert result.exit_code == 0, result.output
    assert "SpaceGet" in result.output


# ---------- members command ----------


def test_members_json(monkeypatch):
    client = MagicMock()
    client.list_space_members.return_value = [
        {"id": "u1", "display_name": "alice", "type": "human", "role": "admin"},
        {"id": "u2", "display_name": "bob", "type": "human", "role": "member"},
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, **kw: "sid")

    result = runner.invoke(app, ["spaces", "members", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 2
    assert data[0]["display_name"] == "alice"


def test_members_text(monkeypatch):
    client = MagicMock()
    client.list_space_members.return_value = {
        "members": [{"id": "u3", "display_name": "carol", "type": "human", "role": "viewer"}]
    }
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, **kw: "sid")

    result = runner.invoke(app, ["spaces", "members", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["display_name"] == "carol"


def test_members_with_explicit_space_id(monkeypatch):
    client = MagicMock()
    client.list_space_members.return_value = [
        {"id": "u4", "display_name": "dave", "type": "human", "role": "admin"}
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)

    result = runner.invoke(app, ["spaces", "members", "explicit-sid", "--json"])
    assert result.exit_code == 0, result.output
    client.list_space_members.assert_called_once_with("explicit-sid")


def test_members_text_table(monkeypatch):
    """Regression for #55: the Member column must render display_name, not
    the obsolete username key. Header must be 'Member' (not 'User') because
    rows include both humans and agents."""
    client = MagicMock()
    client.list_space_members.return_value = [
        {"id": "u5", "display_name": "eve", "type": "human", "role": "member"},
        {"id": "a1", "display_name": "aX", "type": "agent", "role": "member"},
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, **kw: "sid")

    result = runner.invoke(app, ["spaces", "members"])
    assert result.exit_code == 0, result.output
    assert "Member" in result.output
    assert "Type" in result.output
    assert "eve" in result.output
    assert "aX" in result.output
    assert "human" in result.output
    assert "agent" in result.output
    # The pre-fix CLI used header "User"; assert we've moved past it so a
    # future revert is loud.
    assert "User " not in result.output


def test_members_text_table_does_not_use_obsolete_username_key(monkeypatch):
    """Regression for #55: a mock that only supplies the obsolete `username`
    key (no `display_name`) must NOT show that value in the output. This is
    the literal pre-fix shape — confirms we no longer accidentally render it."""
    client = MagicMock()
    client.list_space_members.return_value = [
        {"username": "should-not-render", "role": "admin"},
    ]
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: client)
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, **kw: "sid")

    result = runner.invoke(app, ["spaces", "members"])
    assert result.exit_code == 0, result.output
    assert "should-not-render" not in result.output


# ---------- use_space text output ----------


def test_spaces_use_text_output(monkeypatch):
    class FakeClient:
        def list_spaces(self):
            return {"spaces": [{"id": "s1", "slug": "my-space"}]}

        def whoami(self):
            return {"bound_agent": {"agent_name": "bot", "allowed_spaces": [{"space_id": "s1"}]}}

    saved = {}
    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: FakeClient())
    monkeypatch.setattr("ax_cli.commands.spaces.save_space_id", lambda sid, **kw: saved.update(space_id=sid))
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, explicit=None: "s1")

    result = runner.invoke(app, ["spaces", "use", "my-space"])
    assert result.exit_code == 0, result.output
    assert "Current space" in result.output
    assert "my-space" in result.output


def test_spaces_use_text_warns_unattached(monkeypatch):
    class FakeClient:
        def list_spaces(self):
            return {"spaces": [{"id": "s1", "slug": "my-space"}]}

        def whoami(self):
            return {
                "bound_agent": {
                    "agent_name": "orion",
                    "allowed_spaces": [{"space_id": "other"}],
                }
            }

    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: FakeClient())
    monkeypatch.setattr("ax_cli.commands.spaces.save_space_id", lambda sid, **kw: None)
    monkeypatch.setattr("ax_cli.commands.spaces.resolve_space_id", lambda c, explicit=None: "s1")

    result = runner.invoke(app, ["spaces", "use", "my-space"])
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "orion" in result.output


def test_spaces_use_accepts_slug_and_warns_when_bound_agent_not_attached(monkeypatch):
    saved = {}

    class FakeClient:
        def list_spaces(self):
            return {
                "spaces": [
                    {"id": "private-space", "slug": "madtank-workspace", "name": "madtank's Workspace"},
                    {"id": "team-space", "slug": "ax-cli-dev", "name": "aX CLI Dev"},
                ]
            }

        def whoami(self):
            return {
                "bound_agent": {
                    "agent_name": "orion",
                    "allowed_spaces": [{"space_id": "private-space", "name": "madtank's Workspace"}],
                }
            }

    def fake_save_space_id(space_id, *, local=True):
        saved["space_id"] = space_id
        saved["local"] = local

    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: FakeClient())
    monkeypatch.setattr("ax_cli.commands.spaces.save_space_id", fake_save_space_id)

    result = runner.invoke(app, ["spaces", "use", "ax-cli-dev", "--json"])

    assert result.exit_code == 0, result.output
    assert saved == {"space_id": "team-space", "local": True}
    payload = json.loads(result.output)
    assert payload["space_id"] == "team-space"
    assert payload["space_label"] == "ax-cli-dev"
    assert payload["scope"] == "local"
    assert payload["bound_agent"] == "orion"
    assert payload["bound_agent_allowed"] is False


def test_spaces_use_global_saves_global_config(monkeypatch):
    saved = {}

    class FakeClient:
        def list_spaces(self):
            return {"spaces": [{"id": "team-space", "slug": "ax-cli-dev", "name": "aX CLI Dev"}]}

        def whoami(self):
            return {}

    def fake_save_space_id(space_id, *, local=True):
        saved["space_id"] = space_id
        saved["local"] = local

    monkeypatch.setattr("ax_cli.commands.spaces.get_client", lambda: FakeClient())
    monkeypatch.setattr("ax_cli.commands.spaces.save_space_id", fake_save_space_id)

    result = runner.invoke(app, ["spaces", "use", "ax-cli-dev", "--global", "--json"])

    assert result.exit_code == 0, result.output
    assert saved == {"space_id": "team-space", "local": False}
    assert json.loads(result.output)["scope"] == "global"
