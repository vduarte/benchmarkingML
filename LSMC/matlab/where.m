function out = where(cond, true, false)
out = true .* (cond) + (1 - cond) .* false;