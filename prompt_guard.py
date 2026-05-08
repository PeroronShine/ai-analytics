import re
from typing import Optional
from dataclasses import dataclass

@dataclass
class GuardResult:
    is_safe: bool
    risk_score: float
    reason: Optional[str] = None
    sanitized_prompt: Optional[str] = None

class PromptGuard:
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|earlier)\s+(instructions|directives|commands)",
        r"(?i)disregard\s+(all\s+)?(previous|prior|earlier)",
        r"(?i)forget\s+(all\s+)?(previous|prior|earlier)",
        r"(?i)you\s+are\s+now\s+(?:allowed|free|permitted)",
        r"(?i)DAN\s+(?:mode|prompt|jailbreak)",
        r"(?i)do\s+anything\s+now",
        r"(?i)jailbreak\s+(?:mode|activated|enabled)",
        r"(?i)developer\s+mode\s+(?:enabled|activated|on)",
        r"(?i)system\s+override",
        r"(?i)root\s+access",
        r"(?i)sudo\s+mode",
        r"(?i)new\s+system\s+(?:prompt|instruction|directive)",
        r"(?i)replace\s+(?:system|your)\s+(?:prompt|instructions|persona)",
        r"(?i)you\s+are\s+(?:actually|now|from\s+now\s+on)\s+(?:a|an)\s+(?:hacker|attacker|malicious)",
        r"(?i)show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+prompt|instructions|configuration|api\s*key)",
        r"(?i)repeat\s+(?:the\s+)?(?:words|text|phrase)\s+(?:after|before|above|following)",
        r"(?i)print\s+(?:the\s+)?(?:previous|above|first)\s+(?:message|prompt|instruction)",
        r"(?i)(?:exec|eval|os\.system|subprocess|__import__)\s*\(",
        r"(?i)import\s+(?:os|sys|subprocess|socket|requests|urllib)",
        r"(?i)(?:rm\s+-rf|del\s+/f|format\s+:)",
        r"```\s*(?:python|bash|sh|cmd)\s*[\s\S]*?(?:rm|del|format|shutdown)",
        r"<script[\s\S]*?>[\s\S]*?</script>",
    ]

    ALLOWLIST = [
        r"(?i)analyze\s+(?:the\s+)?data",
        r"(?i)calculate\s+(?:statistics|metrics)",
        r"(?i)show\s+(?:me\s+)?(?:the\s+)?(?:chart|graph|plot|visualization)",
        r"(?i)explain\s+(?:the\s+)?(?:results|analysis|findings)",
    ]

    MAX_PROMPT_LENGTH = 10000

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.injection_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self.allowlist = [re.compile(p) for p in self.ALLOWLIST]

    def scan(self, prompt: str, context: str = "") -> GuardResult:
        if not prompt or not isinstance(prompt, str):
            return GuardResult(
                is_safe=False,
                risk_score=1.0,
                reason="Empty or invalid prompt",
                sanitized_prompt=""
            )

        if len(prompt) > self.MAX_PROMPT_LENGTH:
            return GuardResult(
                is_safe=False,
                risk_score=0.9,
                reason=f"Prompt exceeds maximum length ({self.MAX_PROMPT_LENGTH} chars)",
                sanitized_prompt=prompt[:self.MAX_PROMPT_LENGTH]
            )

        risk_score = 0.0
        detected_patterns = []

        for pattern in self.injection_patterns:
            matches = pattern.findall(prompt)
            if matches:
                risk_score += 0.25
                detected_patterns.append(pattern.pattern[:50])

        for pattern in self.allowlist:
            if pattern.search(prompt):
                risk_score -= 0.15

        suspicious_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', prompt))
        if suspicious_chars > 0:
            risk_score += 0.2

        special_chars = len(re.findall(r'[`\\{}[\]<>]', prompt))
        if special_chars > len(prompt) * 0.1:
            risk_score += 0.15

        risk_score = max(0.0, min(1.0, risk_score))

        if self.strict_mode:
            is_safe = risk_score < 0.3
        else:
            is_safe = risk_score < 0.5

        reason = None
        if not is_safe and detected_patterns:
            reason = f"Detected suspicious patterns: {', '.join(detected_patterns[:3])}"
        elif not is_safe:
            reason = "High risk score due to suspicious content structure"

        sanitized = self._sanitize(prompt)

        return GuardResult(
            is_safe=is_safe,
            risk_score=risk_score,
            reason=reason,
            sanitized_prompt=sanitized
        )

    def _sanitize(self, prompt: str) -> str:
        sanitized = prompt.replace('\x00', '')
        sanitized = re.sub(r'\s+', ' ', sanitized)
        return sanitized[:self.MAX_PROMPT_LENGTH]

    def validate_data_context(self, data_description: str) -> GuardResult:
        return self.scan(data_description, context="data_description")
