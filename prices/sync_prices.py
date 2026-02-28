#!/usr/bin/env python3
"""
Sync model prices from LobeHub TypeScript model definitions and MartialBE upstream.

Data source: https://github.com/sijinhui/lobehub (fork of lobehub/lobe-chat)
TS files at: packages/model-bank/src/aiModels/{provider}.ts
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "requests",
# ]
# ///


import json
import os
import re
import requests

# LobeHub provider name -> one-hub channel_type
LOBEHUB_PROVIDER_MAP = {
    "openai": 1,
#     "azure": 3,
    "anthropic": 14,
    "wenxin": 15,
#     "zhipu": 16,
    "qwen": 17,
#     "spark": 18,
#     "ai360": 19,
#     "tencentcloud": 23,
    "google": 25,
#     "baichuan": 26,
#     "minimax": 27,
    "deepseek": 28,
    "moonshot": 29,
#     "mistral": 30,
    "groq": 31,
#     "bedrock": 32,
#     "zeroone": 33,
#     "cloudflare": 35,
#     "cohere": 36,
#     "ollama": 39,
#     "hunyuan": 40,
#     "vertexai": 42,
#     "siliconcloud": 45,
#     "jina": 47,
#     "github": 49,
#     "replicate": 52,
#     "xai": 56,
#     "openrouter": 20,
}

# LobeHub pricing unit name -> prices.json extra_ratios key
UNIT_TO_EXTRA_KEY = {
    "textInput_cacheRead": "cached_read_tokens",
    "textInput_cacheWrite": "cached_write_tokens",
    "audioInput": "input_audio_tokens",
    "audioOutput": "output_audio_tokens",
    "reasoning": "reasoning_tokens",
}

# Model types to skip (non-token pricing)
SKIP_TYPES = {"image", "tts", "stt", "realtime"}

# CNY to USD approximate rate
CNY_TO_USD = 7.3

TS_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/sijinhui/lobehub/"
    "refs/heads/main/packages/model-bank/src/aiModels/{provider}.ts"
)

BASE_PRICES_URL = (
    "https://fastly.jsdelivr.net/gh/sijinhui/one-hub/prices/prices.json"
)

DOLLAR_RATE = 0.002  # from model/price.go


def usd_to_ratio(usd_per_million: float) -> float:
    """Convert $/M tokens to internal ratio. $2.00/M -> 1.0"""
    return usd_per_million / 1000 / DOLLAR_RATE


def extract_model_blocks(ts_content: str) -> list[str]:
    """Extract top-level model object blocks from TS array content."""
    blocks = []
    depth = 0
    start = -1

    for i, ch in enumerate(ts_content):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(ts_content[start:i + 1])
                start = -1

    return blocks


def extract_field(block: str, field: str) -> str | None:
    """Extract a simple string/value field like id: 'xxx' or type: 'xxx'."""
    # Match both single and double quotes
    m = re.search(rf"\b{field}\s*:\s*['\"]([^'\"]+)['\"]", block)
    return m.group(1) if m else None


def extract_pricing_block(block: str) -> str | None:
    """Extract the pricing: { ... } sub-block from a model block."""
    m = re.search(r'\bpricing\s*:\s*\{', block)
    if not m:
        return None

    start = m.start()
    depth = 0
    for i in range(m.end() - 1, len(block)):
        if block[i] == '{':
            depth += 1
        elif block[i] == '}':
            depth -= 1
            if depth == 0:
                return block[start:i + 1]
    return None


def extract_currency(pricing_block: str) -> str | None:
    """Extract currency field from pricing block."""
    m = re.search(r"currency\s*:\s*['\"](\w+)['\"]", pricing_block)
    return m.group(1) if m else None


def extract_units(pricing_block: str) -> list[dict]:
    """Extract pricing units from the pricing block."""
    units = []

    # Find the units array
    m = re.search(r'units\s*:\s*\[', pricing_block)
    if not m:
        return units

    # Extract each unit object
    units_start = m.end()
    # Find matching ]
    depth = 1
    units_end = units_start
    for i in range(units_start, len(pricing_block)):
        if pricing_block[i] == '[':
            depth += 1
        elif pricing_block[i] == ']':
            depth -= 1
            if depth == 0:
                units_end = i
                break

    units_str = pricing_block[units_start:units_end]

    # Extract each { ... } unit block
    for unit_block in extract_model_blocks(units_str):
        unit = {}

        name = extract_field(unit_block, "name")
        if name:
            unit["name"] = name

        strategy = extract_field(unit_block, "strategy")
        if strategy:
            unit["strategy"] = strategy

        # Extract rate (numeric)
        rate_m = re.search(r'\brate\s*:\s*([\d.]+)', unit_block)
        if rate_m:
            unit["rate"] = float(rate_m.group(1))

        # For lookup strategy, extract prices
        if strategy == "lookup":
            unit["rate"] = _extract_lookup_rate(unit_block)

        if "name" in unit and "rate" in unit:
            units.append(unit)

    return units


def _extract_lookup_rate(unit_block: str) -> float | None:
    """Extract the best rate from a lookup pricing block.
    For TTL-based (e.g. Anthropic cache write), use the shorter TTL (5m) price.
    Otherwise use the first/lowest price found.
    """
    # Try to find prices object: prices: { '5m': 6.25, '1h': 10 }
    prices_m = re.search(r'prices\s*:\s*\{([^}]+)\}', unit_block)
    if not prices_m:
        return None

    prices_str = prices_m.group(1)
    # Extract all key: value pairs
    pairs = re.findall(r"['\"](\w+)['\"]\s*:\s*([\d.]+)", prices_str)
    if not pairs:
        return None

    # Prefer shorter TTL
    ttl_order = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '8h', '24h']
    for ttl in ttl_order:
        for key, val in pairs:
            if key == ttl:
                return float(val)

    # Fallback: return first value
    return float(pairs[0][1])


def parse_provider_ts(provider: str, ts_content: str, channel_type: int) -> list[dict]:
    """Parse a provider's TS file and return price entries."""
    entries = []

    for block in extract_model_blocks(ts_content):
        model_id = extract_field(block, "id")
        model_type = extract_field(block, "type")

        if not model_id:
            continue

        # Skip non-token model types
        if model_type and model_type in SKIP_TYPES:
            continue

        pricing_block = extract_pricing_block(block)
        if not pricing_block:
            continue

        currency = extract_currency(pricing_block)
        units = extract_units(pricing_block)
        if not units:
            continue

        input_rate = None
        output_rate = None
        extra_ratios = {}

        for unit in units:
            name = unit["name"]
            rate = unit["rate"]
            if rate is None:
                continue

            # Convert CNY to USD if needed
            usd_rate = rate
            if currency == "CNY":
                usd_rate = rate / CNY_TO_USD

            ratio = usd_to_ratio(usd_rate)

            if name == "textInput":
                input_rate = ratio
            elif name == "textOutput":
                output_rate = ratio
            elif name in UNIT_TO_EXTRA_KEY:
                extra_ratios[UNIT_TO_EXTRA_KEY[name]] = ratio
                # Backward compat: cached_read_tokens also writes cached_tokens
                if name == "textInput_cacheRead":
                    extra_ratios["cached_tokens"] = ratio

        if input_rate is None or output_rate is None:
            continue

        entry = {
            "model": model_id,
            "type": "tokens",
            "channel_type": channel_type,
            "input": round(input_rate, 4),
            "output": round(output_rate, 4),
        }

        if extra_ratios:
            entry["extra_ratios"] = {
                k: round(v, 4) for k, v in extra_ratios.items()
            }

        entries.append(entry)

    return entries


def fetch_lobehub_prices() -> list[dict]:
    """Fetch and parse prices from all mapped LobeHub providers."""
    all_entries = []

    for provider, channel_type in LOBEHUB_PROVIDER_MAP.items():
        url = TS_URL_TEMPLATE.format(provider=provider)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"  跳过 {provider}: 文件不存在")
                continue
            resp.raise_for_status()
            ts_content = resp.text
        except requests.RequestException as e:
            print(f"  获取 {provider} 失败: {e}")
            continue

        entries = parse_provider_ts(provider, ts_content, channel_type)
        print(f"  {provider}: 解析到 {len(entries)} 个模型价格")
        all_entries.extend(entries)

    return all_entries


def fetch_base_prices() -> list[dict]:
    """Fetch base prices from MartialBE upstream."""
    resp = requests.get(BASE_PRICES_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def sync_prices():
    print("从 LobeHub 获取价格数据...")
    lobehub_prices = fetch_lobehub_prices()
    print(f"共解析到 {len(lobehub_prices)} 个 LobeHub 价格条目")

    print("\n从 MartialBE 获取基础价格...")
    base_prices = fetch_base_prices()
    print(f"基础价格条目: {len(base_prices)}")

    # Build lookup: (model, channel_type) -> entry from lobehub
    lobehub_lookup = {}
    for entry in lobehub_prices:
        key = (entry["model"], entry["channel_type"])
        lobehub_lookup[key] = entry

    # Merge: lobehub overrides base
    merged = []
    used_keys = set()

    for base_entry in base_prices:
        key = (base_entry["model"], base_entry.get("channel_type", 0))
        if key in lobehub_lookup:
            merged.append(lobehub_lookup[key])
            used_keys.add(key)
        else:
            merged.append(base_entry)

    # Add lobehub entries not in base
    for key, entry in lobehub_lookup.items():
        if key not in used_keys:
            merged.append(entry)

    # Write output
    os.makedirs("prices", exist_ok=True)
    with open("prices/prices.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n写入 prices/prices.json: {len(merged)} 条记录")
    print("价格数据同步成功！")


if __name__ == "__main__":
    sync_prices()
