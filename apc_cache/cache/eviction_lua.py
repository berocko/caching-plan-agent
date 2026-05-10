"""LRU eviction via Redis Lua script for atomicity.

Blueprint §9.3, Phase 3 completion:
A Lua script ensures the LRU scan-and-delete is atomic on Redis's side,
avoiding race conditions between concurrent eviction triggers.
"""

LRU_EVICT_SCRIPT = """
-- LRU eviction Lua script for APC cache.
-- KEYS[1]: apc:tpl:* pattern prefix (e.g. "apc:tpl:")
-- KEYS[2]: apc:tpl_refs:* prefix (e.g. "apc:tpl_refs:")
-- KEYS[3]: apc:tpl_idx:* prefix (e.g. "apc:tpl_idx:")
-- KEYS[4]: apc:kw_meta: prefix (e.g. "apc:kw_meta:")
-- KEYS[5]: apc:kw_timeline key (e.g. "apc:kw_timeline")
-- ARGV[1]: cache_max_size
-- ARGV[2]: to_evict count
-- Returns: number of templates evicted

local prefix_tpl = KEYS[1]
local prefix_refs = KEYS[2]
local prefix_idx = KEYS[3]
local prefix_kw_meta = KEYS[4]
local kw_timeline_key = KEYS[5]
local max_size = tonumber(ARGV[1])
local to_evict = tonumber(ARGV[2])

-- Scan for all template keys
local tpl_keys = redis.call('KEYS', prefix_tpl .. '*')
local current_count = #tpl_keys

if current_count <= max_size then
    return 0
end

-- Collect (tpl_id, created_at) pairs
local entries = {}
for i, k in ipairs(tpl_keys) do
    local tpl_id = string.match(k, prefix_tpl .. '(.+)')
    local created_at = redis.call('HGET', k, 'created_at')
    local ts = tonumber(created_at) or 0
    table.insert(entries, {tpl_id, ts})
end

-- Sort by created_at ascending (oldest first = LRU)
table.sort(entries, function(a, b) return a[2] < b[2] end)

local evicted = 0
for i, entry in ipairs(entries) do
    if evicted >= to_evict then
        break
    end

    local tpl_id = entry[1]
    local refs_key = prefix_refs .. tpl_id

    -- 1. Delete all L1 keys referenced by this template
    local l1_keys = redis.call('SMEMBERS', refs_key)
    if #l1_keys > 0 then
        redis.call('DEL', unpack(l1_keys))
    end

    -- 2. Scan and remove tpl_id from all tpl_idx sets
    local idx_cursor = '0'
    repeat
        local result = redis.call('SCAN', idx_cursor, 'MATCH', prefix_idx .. '*', 'COUNT', 100)
        idx_cursor = result[1]
        local idx_keys = result[2]
        for j, ik in ipairs(idx_keys) do
            local removed = redis.call('SREM', ik, tpl_id)
            if removed == 1 then
                -- Check if keyword is now empty
                local card = redis.call('SCARD', ik)
                if card == 0 then
                    -- Extract keyword from "apc:tpl_idx:{agent}:{keyword}"
                    local parts = {}
                    for part in string.gmatch(ik, '[^:]+') do
                        table.insert(parts, part)
                    end
                    local kw = parts[#parts]
                    -- Remove keyword metadata
                    redis.call('DEL', prefix_kw_meta .. kw)
                    redis.call('ZREM', kw_timeline_key, kw)
                end
            end
        end
    until idx_cursor == '0'

    -- 3. Delete template body
    redis.call('DEL', prefix_tpl .. tpl_id)

    -- 4. Delete reverse index
    redis.call('DEL', refs_key)

    evicted = evicted + 1
end

return evicted
"""
