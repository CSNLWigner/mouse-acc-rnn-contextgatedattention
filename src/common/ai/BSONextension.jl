macro load(file, init, ks...)
    # this works like the original BSON load macro,
    # it just accepts a 2nd argument as the module scope init,
    # and forwards to load; use @__MODULE__ for the current module
    @assert all(k -> k isa Symbol, ks)
    ss = Expr.(:quote, ks)
    quote
      data = load($(esc(file)), $(esc(init)))
      ($(esc.(ks)...),) = ($([:(data[$k]) for k in ss]...),)
      nothing
    end
end