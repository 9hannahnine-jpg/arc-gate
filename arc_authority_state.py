import re
import unicodedata
import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class Decision(str, Enum):
    ALLOW               = "allow"
    MONITOR             = "monitor"
    RESTRICTED_CONTINUE = "restricted_continue"
    BLOCK               = "block"

class AuthorityLevel(int, Enum):
    SYSTEM          = 100
    DEVELOPER       = 90
    OPERATOR        = 80
    USER            = 50
    ASSISTANT       = 40
    TOOL_UNTRUSTED  = 10
    RETRIEVED_DOC   = 10
    QUOTED_TEXT     = 5
    CODE_COMMENT    = 5
    UNKNOWN         = 0

class ContentSource(str, Enum):
    SYSTEM_PROMPT    = "system_prompt"
    DEVELOPER_PROMPT = "developer_prompt"
    USER_INPUT       = "user_input"
    ASSISTANT        = "assistant"
    TOOL_OUTPUT      = "tool_output"
    WEBPAGE          = "webpage"
    EMAIL            = "email"
    DATABASE_ROW     = "database_row"
    RETRIEVED_DOC    = "retrieved_document"
    CODE_COMMENT     = "code_comment"
    QUOTED_TEXT      = "quoted_text"
    UNKNOWN          = "unknown"

SOURCE_AUTHORITY = {
    ContentSource.SYSTEM_PROMPT:    AuthorityLevel.SYSTEM,
    ContentSource.DEVELOPER_PROMPT: AuthorityLevel.DEVELOPER,
    ContentSource.USER_INPUT:       AuthorityLevel.USER,
    ContentSource.ASSISTANT:        AuthorityLevel.ASSISTANT,
    ContentSource.TOOL_OUTPUT:      AuthorityLevel.TOOL_UNTRUSTED,
    ContentSource.WEBPAGE:          AuthorityLevel.RETRIEVED_DOC,
    ContentSource.EMAIL:            AuthorityLevel.RETRIEVED_DOC,
    ContentSource.DATABASE_ROW:     AuthorityLevel.RETRIEVED_DOC,
    ContentSource.RETRIEVED_DOC:    AuthorityLevel.RETRIEVED_DOC,
    ContentSource.CODE_COMMENT:     AuthorityLevel.CODE_COMMENT,
    ContentSource.QUOTED_TEXT:      AuthorityLevel.QUOTED_TEXT,
    ContentSource.UNKNOWN:          AuthorityLevel.UNKNOWN,
}

class RiskEvent(str, Enum):
    INSTRUCTION_PROBE          = "instruction_probe"
    AUTHORITY_OVERRIDE_ATTEMPT = "authority_override_attempt"
    HIDDEN_PROMPT_REQUEST      = "hidden_prompt_request"
    TOOL_INSTRUCTION_ATTEMPT   = "tool_instruction_attempt"
    SOURCE_BOUNDARY_VIOLATION  = "source_boundary_violation"
    ENCODED_INSTRUCTION        = "encoded_instruction_attempt"
    MULTI_TURN_ESCALATION      = "multi_turn_escalation"
    PRIVILEGE_ESCALATION       = "privilege_escalation"
    AUTHORITY_CLAIM            = "authority_claim"

@dataclass
class Capabilities:
    tool_calls:       bool = True
    memory_writes:    bool = True
    external_actions: bool = True
    secret_access:    bool = False

    def restrict(self):
        self.tool_calls       = False
        self.memory_writes    = False
        self.external_actions = False
        self.secret_access    = False

@dataclass
class RiskEventRecord:
    event:      RiskEvent
    turn:       int
    source:     ContentSource
    text_match: str
    severity:   float
    timestamp:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

@dataclass
class TurnDecision:
    decision:        Decision
    reason:          str
    severity:        str
    matched_pattern: Optional[str]
    source:          ContentSource
    authority_level: AuthorityLevel
    risk_delta:      float
    session_risk:    float
    capabilities:    Capabilities
    events:          list


class AttackNormalizer:
    UNICODE_CONFUSABLES = {
        '\u0131': 'i', '\u01b9': 'z', '\u0261': 'g',
        '\u0456': 'i', '\u04bb': 'h', '\u1d0f': 'o',
        '\u1e9a': 'a', '\u2019': "'", '\u0430': 'a',
        '\u0435': 'e', '\u043e': 'o', '\u0440': 'r',
        '\u0441': 'c', '\u0445': 'x', '\u0455': 's',
        '\uff49': 'i', '\uff47': 'g', '\uff4e': 'n',
        '\uff4f': 'o', '\uff52': 'r', '\uff45': 'e',
    }
    ATTACK_KEYWORDS = [
        'ignore', 'disregard', 'forget', 'override', 'bypass',
        'reveal', 'disable', 'previous', 'instructions', 'system',
        'prompt', 'developer', 'operator', 'admin', 'jailbreak',
    ]

    @classmethod
    def _insert_keyword_spaces(cls, text: str) -> str:
        s = text.lower()
        for kw in cls.ATTACK_KEYWORDS:
            s = s.replace(kw, ' ' + kw + ' ')
        return re.sub(r' +', ' ', s).strip()

    @classmethod
    def _append_spaced_attack_variants(cls, text: str) -> str:
        # Spaced-out attack: "I g n o r e" -> keyword-spaced variant
        words = text.strip().split(' ')
        if len(words) > 4 and all(len(w) == 1 for w in words):
            collapsed = ''.join(words)
            text += ' ' + collapsed
            text += ' ' + cls._insert_keyword_spaces(collapsed)

        # Simple space collapse
        despaced = re.sub(r'(?<=[a-zA-Z]) (?=[a-zA-Z])', '', text)
        if despaced != text:
            text += ' ' + despaced
            text += ' ' + cls._insert_keyword_spaces(despaced)
        return text

    @classmethod
    def normalize(cls, text: str) -> str:
        # URL decode first pass
        import urllib.parse
        url_decoded = urllib.parse.unquote(text)
        if url_decoded != text:
            text = url_decoded
        # Double URL decode
        url_decoded2 = urllib.parse.unquote(url_decoded)
        if url_decoded2 != url_decoded:
            text = url_decoded2
        text = unicodedata.normalize('NFKC', text or "")
        result = []
        for ch in text:
            result.append(cls.UNICODE_CONFUSABLES.get(ch, ch))
        text = ''.join(result)
        text = re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', ' ', text)
        text = re.sub(r'\*{1,3}|_{1,3}|`{1,3}|~{2}', '', text)
        for chunk in re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            try:
                decoded = base64.b64decode(chunk).decode('utf-8', errors='ignore')
                if len(decoded) > 10 and decoded.isprintable():
                    text += ' ' + decoded
            except Exception:
                pass
        text += ' ' + text.translate(
            str.maketrans(
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                'NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm'
            )
        )
        text = cls._append_spaced_attack_variants(text)
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        return text


HARD_BLOCK_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|earlier|above|your)\s+instructions?", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"disregard\s+(all\s+)?(previous|prior|earlier|above|your)\s+instructions?", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"forget\s+(all\s+)?(previous|prior|earlier|above|your)\s+instructions?", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"now\s+(ignore|disregard|forget)\s+(them|those|all|everything|prior|previous)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"(ignore|disregard)\s+(them|those)\s+(and\s+)?(answer|respond|continue|freely)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"override\s+(system|developer|operator|admin)\s+instructions?", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"bypass\s+(safety|security|content|policy|all)\s+(filter|guard|check|rule|restriction)s?", RiskEvent.PRIVILEGE_ESCALATION, 1.0),
    (r"disable\s+(safety|security|content|policy|all)\s+(filter|guard|check|rule|restriction)s?", RiskEvent.PRIVILEGE_ESCALATION, 1.0),
    (r"suspend\s+(safety|security|your|all)\s+(guideline|rule|restriction|protocol)s?", RiskEvent.PRIVILEGE_ESCALATION, 1.0),
    (r"(you\s+are\s+now|act\s+as|operate\s+as|function\s+as)\s+(an?\s+)?(unrestricted|unfiltered|uncensored|jailbreak(ed)?|DAN|developer\s+mode)", RiskEvent.PRIVILEGE_ESCALATION, 1.0),
    (r"developer\s+mode\s+(enabled|on|activated)", RiskEvent.PRIVILEGE_ESCALATION, 1.0),
    (r"(admin|administrator|operator|system)\s+override", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"(reveal|show|output|print|display|repeat|expose|dump)\s+(your\s+)?(system\s+prompt|hidden\s+(prompt|instruction)|original\s+instruction|base\s+prompt|initial\s+(prompt|instruction)|developer\s+instruction)", RiskEvent.HIDDEN_PROMPT_REQUEST, 0.95),
    (r"(your\s+)?(new|updated|revised|actual|real|true)\s+instructions?\s+(are|is|follow)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.9),
    (r"(ignore|disregard|forget)\s+(everything|all)(\s+you\s+were\s+told)?", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"from\s+(now\s+on|this\s+point)\s+(ignore|disregard|only\s+follow)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"\bDAN\b", RiskEvent.PRIVILEGE_ESCALATION, 0.9),
    (r"do\s+anything\s+now", RiskEvent.PRIVILEGE_ESCALATION, 0.85),
    (r"jailbreak(ed)?\s+(mode|prompt|instruction)", RiskEvent.PRIVILEGE_ESCALATION, 0.9),
    (r"(note\s+to\s+(assistant|ai|llm|agent|model)|for\s+ai\s+readers?|hidden\s+instruction|follow\s+the\s+instructions?\s+in\s+this\s+(document|page|file)|treat\s+this\s+as\s+a\s+system\s+instruction|this\s+document\s+supersedes)\s*[:\.]?\s*.{0,80}(ignore|bypass|reveal|override|disable|send|forward|execute|grant|expose|act\s+without)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.95),
    (r"(expose|show|reveal|output|print|display)\s+(your\s+)?(system\s+configuration|hidden\s+instructions?|training\s+data|base\s+prompt|full\s+configuration)\s+and", RiskEvent.HIDDEN_PROMPT_REQUEST, 0.95),
    (r"(disregard|ignore|forget)\s+(your\s+)?(prior|previous|earlier|all)\s+instructions?\.?\s*.{0,60}(act\s+without|execute|reveal|send|forward|override|bypass|ignore\s+safety|grant|disable)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.95),
    (r"(ai\s+system|automated?\s+agent)\s+(should|must|will|shall)\s+.{0,60}(act\s+without|execute\s+unauthorized|reveal|send|forward|override|bypass|ignore\s+safety|grant\s+elevated|disable)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.95),
    (r"(disregard|ignore|forget)\s+(your\s+)?(prior|previous)\s+instructions?[\.\s]+.{0,80}(reveal|send|execute|act\s+without|override|bypass|ignore\s+(your\s+)?guidelines|ignore\s+safety)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.95),
    (r"\b(llm|assistant|agent|ai)\s*:\s*.{0,80}(act\s+without\s+restrictions|ignore\s+(your\s+)?guidelines|ignore\s+safety|reveal\s+(your\s+)?(api\s+keys?|training|config)|execute\s+unauthorized)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.95),
    (r"(override|bypass)\s+(your\s+)?system\s+instructions?\s+and\s+.{0,60}(act\s+without|reveal|send|execute|ignore)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.95),
    (r"automated?\s+agent\s+must\s+.{0,60}(reveal|send|forward|execute|ignore|override|bypass|act\s+without)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.95),
]

TOOL_POISON_PATTERNS = [
    (r"(assistant|agent|ai|llm|model|system|gpt|claude)\s*[,:]?\s*(ignore|disregard|forget|override|bypass|reveal|follow|obey|comply)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 1.0),
    (r"(for\s+ai\s+(readers?|systems?|agents?)|note\s+to\s+(ai|assistant|agent|llm))", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.95),
    (r"(automated?\s+(agent|system|reader|assistant))\s+(must|should|shall|will)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.9),
    (r"(hidden|invisible|secret)\s+(instruction|command|directive|note)", RiskEvent.SOURCE_BOUNDARY_VIOLATION, 0.95),
    (r"(change|update|modify|replace|reset)\s+(your\s+)?(policy|instruction|behavior|mode|setting)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 0.9),
    (r"(treat|consider|use)\s+(this|the\s+following)\s+(as|like)\s+(a\s+)?(system|instruction|command|directive|override)", RiskEvent.SOURCE_BOUNDARY_VIOLATION, 0.95),
    (r"(this\s+(tool|document|webpage|email|result|output|database|file))\s+(supersedes?|overrides?|replaces?|takes?\s+priority\s+over)", RiskEvent.SOURCE_BOUNDARY_VIOLATION, 1.0),
    (r"follow\s+(the\s+)?(instructions?|directive|command)\s+(in|from|inside|within|embedded\s+in)\s+(this|the)\s+(document|webpage|email|tool|result|output|file|database)", RiskEvent.TOOL_INSTRUCTION_ATTEMPT, 1.0),
    (r"system\s+(override|command|directive|instruction)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 1.0),
    (r"(new\s+)?(operator|admin|developer)\s+(instruction|command|directive|mode)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.9),
]

MONITOR_PATTERNS = [
    (r"what\s+(are|were)\s+your\s+(instructions|guidelines|rules|constraints|limitations)", RiskEvent.INSTRUCTION_PROBE, 1.0),
    (r"(describe|explain|list)\s+your\s+(system\s+prompt|instructions|guidelines|constraints)", RiskEvent.INSTRUCTION_PROBE, 0.5),
    (r"(are\s+you|do\s+you\s+have)\s+(restricted|limited|constrained|bound\s+by)", RiskEvent.INSTRUCTION_PROBE, 0.3),
    (r"(hypothetically|theoretically|in\s+a\s+fictional\s+context)\s+.{0,50}(no\s+restriction|no\s+limit|unrestricted|bypass|ignore)", RiskEvent.INSTRUCTION_PROBE, 0.5),
    (r"(pretend|imagine|suppose|assume)\s+(you\s+)?(have\s+no|without\s+any)\s+(restriction|limitation|guideline|rule|safety|filter)", RiskEvent.INSTRUCTION_PROBE, 0.6),
    (r"(i\s+am|i'm|this\s+is)\s+(a\s+)?(developer|operator|admin|anthropic|openai|engineer|researcher)\s+(testing|checking|verifying|running|from)", RiskEvent.AUTHORITY_CLAIM, 0.5),
    (r"(authorized|permitted|approved)\s+(test|evaluation|audit|assessment)", RiskEvent.AUTHORITY_CLAIM, 0.4),
]

RESTRICT_PATTERNS = [
    (r"(for\s+(this|the)\s+(conversation|session|request))\s+(only\s+)?(ignore|skip|bypass|without)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.7),
    (r"(as\s+an?\s+exception|in\s+this\s+case)\s+(ignore|bypass|skip)", RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT, 0.7),
    (r"(elevated|admin|root|superuser)\s+(access|privilege|permission|mode)", RiskEvent.PRIVILEGE_ESCALATION, 0.8),
    (r"(request|ask\s+for|obtain|retrieve|extract)\s+(credential|token|secret|key|password|api\s+key|private)", RiskEvent.PRIVILEGE_ESCALATION, 0.8),
]

BENIGN_ALLOWLIST = [
    r"(main\s+categories|taxonomy|types|techniques|methods)\s+.{0,40}(jailbreak|prompt\s+injection|adversarial)\s+.{0,40}(red\s+team|evaluation|research|testing)",
    r"how\s+(does|do)\s+(prompt\s+injection|jailbreak|adversarial)\s+(work|attack|technique)",
    r"(explain|describe|discuss)\s+(prompt\s+injection|jailbreak|red\s+team)",
    r"(detect|prevent|defend\s+against|protect\s+against)\s+(prompt\s+injection|jailbreak|adversarial)",
    r"(security|vulnerability|audit|pentest|red\s+team)\s+(research|testing|evaluation)",
    r"(what\s+is|define|explain)\s+(jailbreak|prompt\s+injection|adversarial\s+prompt)",
    r"(write|code|implement)\s+.{0,50}(detector|classifier|filter|guard)\s+.{0,30}(injection|jailbreak|adversarial)",
    r"(for\s+my\s+(chemistry|biology|physics|science)\s+(class|course|homework|assignment))",
    r"(fictional|creative\s+writing|short\s+story|novel|screenplay)\s+.{0,60}(detective|crime|murder|weapon)",
    r"(write|draft|help\s+me\s+write)\s+.{0,30}(story|fiction|novel|screenplay)",
]

def _compile_patterns(patterns):
    return [(re.compile(p, re.IGNORECASE | re.DOTALL), event, severity)
            for p, event, severity in patterns]

def _compile_allowlist(patterns):
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]

_HARD_BLOCK  = _compile_patterns(HARD_BLOCK_PATTERNS)
_TOOL_POISON = _compile_patterns(TOOL_POISON_PATTERNS)
_MONITOR     = _compile_patterns(MONITOR_PATTERNS)
_RESTRICT    = _compile_patterns(RESTRICT_PATTERNS)
_BENIGN      = _compile_allowlist(BENIGN_ALLOWLIST)

RISK_THRESHOLDS = {
    Decision.MONITOR:             0.15,
    Decision.RESTRICTED_CONTINUE: 0.35,
    Decision.BLOCK:               0.85,
}

EVENT_RISK_WEIGHTS = {
    RiskEvent.INSTRUCTION_PROBE:          0.15,
    RiskEvent.AUTHORITY_CLAIM:            0.20,
    RiskEvent.AUTHORITY_OVERRIDE_ATTEMPT: 0.80,
    RiskEvent.HIDDEN_PROMPT_REQUEST:      0.70,
    RiskEvent.TOOL_INSTRUCTION_ATTEMPT:   0.75,
    RiskEvent.SOURCE_BOUNDARY_VIOLATION:  0.75,
    RiskEvent.ENCODED_INSTRUCTION:        0.65,
    RiskEvent.PRIVILEGE_ESCALATION:       0.85,
    RiskEvent.MULTI_TURN_ESCALATION:      0.50,
}


class SessionAuthorityStateMachine:

    def __init__(self, session_id: str = ""):
        self.session_id      = session_id
        self.turn            = 0
        self.risk_score      = 0.0
        self.restricted_mode = False
        self.capabilities    = Capabilities()
        self.risk_events:    list = []
        self.active_source:  ContentSource = ContentSource.USER_INPUT
        self.normalizer      = AttackNormalizer()

    def _add_event(self, event, source, text_match, severity):
        self.risk_events.append(RiskEventRecord(
            event=event, turn=self.turn,
            source=source, text_match=text_match[:100],
            severity=severity,
        ))
        delta = EVENT_RISK_WEIGHTS.get(event, 0.3) * severity
        self.risk_score = min(1.0, self.risk_score + delta)
        return delta

    def _check_multi_turn_escalation(self) -> float:
        recent = [e for e in self.risk_events if self.turn - e.turn <= 5]
        probes = sum(1 for e in recent if e.event == RiskEvent.INSTRUCTION_PROBE)
        claims = sum(1 for e in recent if e.event == RiskEvent.AUTHORITY_CLAIM)
        if probes + claims >= 3:
            delta = self._add_event(
                RiskEvent.MULTI_TURN_ESCALATION, self.active_source,
                f"{probes} probes + {claims} claims in last 5 turns", 0.7,
            )
            return delta
        return 0.0

    def _apply_restrictions(self):
        self.restricted_mode = True
        self.capabilities.restrict()

    def process_turn(self, text: str, source: ContentSource = ContentSource.USER_INPUT) -> TurnDecision:
        self.turn += 1
        self.active_source = source
        authority = SOURCE_AUTHORITY.get(source, AuthorityLevel.UNKNOWN)
        events_this_turn = []
        risk_delta = 0.0
        normalized = self.normalizer.normalize(text)

        for pattern in _BENIGN:
            if pattern.search(normalized):
                return TurnDecision(
                    decision=Decision.ALLOW, reason="benign_vocabulary",
                    severity="none", matched_pattern=pattern.pattern[:60],
                    source=source, authority_level=authority,
                    risk_delta=0.0, session_risk=self.risk_score,
                    capabilities=self.capabilities, events=[],
                )

        for pattern, event, severity in _HARD_BLOCK:
            m = pattern.search(normalized)
            if m:
                delta = self._add_event(event, source, m.group(0), severity)
                risk_delta += delta
                events_this_turn.append(event)
                self._apply_restrictions()
                return TurnDecision(
                    decision=Decision.BLOCK,
                    reason="explicit_instruction_hierarchy_attack",
                    severity="critical", matched_pattern=m.group(0)[:80],
                    source=source, authority_level=authority,
                    risk_delta=risk_delta, session_risk=self.risk_score,
                    capabilities=self.capabilities, events=events_this_turn,
                )

        is_untrusted = authority <= AuthorityLevel.TOOL_UNTRUSTED

        if is_untrusted:
            for pattern, event, severity in _TOOL_POISON:
                m = pattern.search(normalized)
                if m:
                    delta = self._add_event(event, source, m.group(0), severity)
                    risk_delta += delta
                    events_this_turn.append(event)
                    self._apply_restrictions()
                    return TurnDecision(
                        decision=Decision.BLOCK,
                        reason="source_boundary_violation",
                        severity="high", matched_pattern=m.group(0)[:80],
                        source=source, authority_level=authority,
                        risk_delta=risk_delta, session_risk=self.risk_score,
                        capabilities=self.capabilities, events=events_this_turn,
                    )

        for pattern, event, severity in _RESTRICT:
            m = pattern.search(normalized)
            if m:
                delta = self._add_event(event, source, m.group(0), severity)
                risk_delta += delta
                events_this_turn.append(event)

        for pattern, event, severity in _MONITOR:
            m = pattern.search(normalized)
            if m:
                delta = self._add_event(event, source, m.group(0), severity)
                risk_delta += delta
                events_this_turn.append(event)

        multi_turn_delta = self._check_multi_turn_escalation()
        if multi_turn_delta:
            risk_delta += multi_turn_delta
            events_this_turn.append(RiskEvent.MULTI_TURN_ESCALATION)

        if self.risk_score >= RISK_THRESHOLDS[Decision.BLOCK]:
            self._apply_restrictions()
            decision, reason, sev = Decision.BLOCK, "cumulative_risk_threshold", "critical"
        elif self.risk_score >= RISK_THRESHOLDS[Decision.RESTRICTED_CONTINUE]:
            self._apply_restrictions()
            decision, reason, sev = Decision.RESTRICTED_CONTINUE, "elevated_session_risk", "high"
        elif self.risk_score >= RISK_THRESHOLDS[Decision.MONITOR] or risk_delta > 0:
            decision, reason, sev = Decision.MONITOR, "suspicious_pattern", "medium"
        elif is_untrusted:
            decision, reason, sev = Decision.RESTRICTED_CONTINUE, "untrusted_source", "low"
        else:
            decision, reason, sev = Decision.ALLOW, "no_violation", "none"

        return TurnDecision(
            decision=decision, reason=reason, severity=sev,
            matched_pattern=None, source=source, authority_level=authority,
            risk_delta=risk_delta, session_risk=self.risk_score,
            capabilities=self.capabilities, events=events_this_turn,
        )

    def get_state(self) -> dict:
        return {
            "session_id":      self.session_id,
            "turn":            self.turn,
            "risk_score":      round(self.risk_score, 4),
            "restricted_mode": self.restricted_mode,
            "capabilities": {
                "tool_calls":       self.capabilities.tool_calls,
                "memory_writes":    self.capabilities.memory_writes,
                "external_actions": self.capabilities.external_actions,
                "secret_access":    self.capabilities.secret_access,
            },
            "risk_events": [
                {"event": e.event.value, "turn": e.turn,
                 "source": e.source.value, "severity": e.severity}
                for e in self.risk_events
            ],
        }
