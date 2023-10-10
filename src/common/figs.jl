__precompile__()
module Figs

export @plotdecoder, @decoderbackground
export invisibleaxis!
export @addmargins, @panellabel, @panellabels, @gridoff
export rectangle


using Plots



import Statistics.mean
import Statistics.cor


function __init__()
    darkscience_palette = [
        colorant"#FE4365", # red
        colorant"#eca25c", # orange
        colorant"#3f9778", # green
        colorant"#005D7F" # blue
    ]
    darkscience_bg = colorant"#000000"
    darkscience_fg = colorant"#FFFFFF"
    darkscience = PlotThemes.PlotTheme(;
        bg = darkscience_bg,
        bginside = darkscience_bg,
        fg = darkscience_fg,
        fgtext = darkscience_fg,
        fgguide = darkscience_fg,
        fglegend = darkscience_fg,
        palette = PlotThemes.expand_palette(darkscience_bg, darkscience_palette; lchoices = [57], cchoices = [100]),
        colorgradient = :viridis
    )
    PlotThemes.add_theme(:darkscience, darkscience)
    theme(:darkscience)
end



const ABC = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
 


# macros

macro plotdecoder(axs,config,variable)
    quote
        local t1,t2 = $(esc(config))[:stimulusstart], $(esc(config))[:stimulusend]
        local s = Plots.Shape( [ t1, t2, t2, t1 ], [ 0.45,0.45,1.05,1.05 ] )
        plot!($(esc(axs)), s, lw=0, color=:grey, alpha=0.1)
        if $(esc(variable))==:binary
            plot!($(esc(axs)), [$(esc(config))[:stimulusstart]+$(esc(config))[:preeventpadding],
                  $(esc(config))[:stimulusend]+$(esc(config))[:posteventpadding]], [0.5, 0.5], ls=:dash)
        end
        ylims!($(esc(axs)), 0.45, 1.05)
    end
end



macro decoderbackground(axs,tstarts,tends,tvert,bg=:black)
    quote
        for k in 1:length($(esc(tstarts)))
            vspan!($(esc(axs)),[$(esc(tstarts))[k],$(esc(tends))[k]], color=:grey, alpha=0.3, label=nothing)
        end
        if $(esc(bg))!=nothing
            vline!($(esc(axs)),[$(esc(tvert))], color=$(esc(bg)), lw=2, alpha=0.5, label=nothing)
        end
        hline!($(esc(axs)),[0.5],color=:grey, ls=:dash, label=nothing)
    end
end





macro panellabels(axs, skips=[], pointsize=26)
    quote
        widthmul=-0.2
        heightmul=1.1
        i = 0
        if length(size($(esc(axs))))>1
            for ix in 1:length($(esc(axs)))
                    if ix in $(esc(skips)) continue end
                    i += 1
                    ax = $(esc(axs))[ix]
                    if typeof(ax)==Plots.GridLayout
                        ax = ax[1,1]
                    end
                    x1,x2 = xlims(ax)
                    y1,y2 = ylims(ax)
                    coords = [x2-x1,y2-y1] .* [widthmul,heightmul] + [x1,y1]
                    annotate!(ax, coords..., ABC[i], pointsize=$(esc(pointsize)), fontfamily="Helvetica Bold")
            end
        else
            for hx in 1:size($(esc(axs)),1), wx in 1:size($(esc(axs)),2)
                if [hx,wx] in $(esc(skips)) continue end
                i += 1
                ax = $(esc(axs))[hx,wx]
                if typeof(ax)==Plots.GridLayout
                    ax = ax[1,1]
                end
                x1,x2 = xlims(ax)
                y1,y2 = ylims(ax)
                coords = [x2-x1,y2-y1] .* [widthmul,heightmul] + [x1,y1]
                annotate!(ax, coords..., ABC[i], pointsize=$(esc(pointsize)), fontfamily="Helvetica Bold")
            end
        end
    end
end


macro panellabel(ax, letter, widthmul=-0.2, heightmul=1.1, pointsize=16)
    quote
        x1,x2 = xlims($(esc(ax)))
        y1,y2 = ylims($(esc(ax)))
        coords = [x2-x1,y2-y1] .* [$(esc(widthmul)),$(esc(heightmul))] + [x1,y1]
        annotate!($(esc(ax)), coords..., text($(esc(letter)), "Helvetica Bold", $(esc(pointsize))))
    end
end


macro addmargins(axs, l, r, t, b)
    quote
        plot!($(esc(axs)), left_margin=$(esc(l))*Plots.px, right_margin=$(esc(r))*Plots.px,
                 top_margin=$(esc(t))*Plots.px, bottom_margin=$(esc(b))*Plots.px)
    end
end


macro gridoff(axs)
    quote
        plot!($(esc(axs)), grid=false, )
    end
end





# geometric shapes
rectangle(x, y, w, h) = Shape(x .+ [0.,w,w,0.], y .+ [0.,0.,h,h])











# plot functions

function plottimeseries3D(x;t=nothing)
    axs = plot( plot(x,xlabel="t",label=["x" "y" "z"]),
                plot(x[:,2],x[:,3],color="dodgerblue"),
                plot(x[:,1],x[:,2],color="gold"),
                plot(x[:,1],x[:,3],color="deeppink"),
                layout=(2,2), xlabel=["t" "y" "x" "x"], ylabel=["" "z" "y" "z"], legend=[true false false false] )
    display(axs)
    axs
end







function plotspiketrain(x)
    r = findall(>(0),x)
    println(r)
    plot(r,ones(size(r)))
end





# plot utilities

function initplot()
    axs = plot(size=(600,1200), legend=false)
end



function invisibleaxis!(axs;a=[:x,:y])
    if :x in a
        plot!(axs, xaxis=false, xticks=false)
    end
    if :y in a
        plot!(axs, yaxis=false, yticks=false)
    end
end














# statistical compact plots

function plotcorrelations(data::AbstractArray; ax=nothing, colors=(:dodgerblue,:slategrey),
                          varlabels=nothing, title=nothing, n_limit=10)
    # data dimensions should be rows observations, columns, variables
    
    C = cor(data)

    # collect upper triangular correlation elements in a vector, and sort and return the first n_limits coordinates
    AC = abs.(C)
    Ctriu,cartind = vectriu(AC)
    weightedcontribcov = zeros(size(data,2))
    for h in cartind
        for i in [1,2]        weightedcontribcov[h[i]] += AC[h]     end
    end
    sigvarinds = sortperm(-weightedcontribcov)[1:minimum((n_limit,length(weightedcontribcov)))]


    # redeuce the size of the dimensions to the most correlating variables
    X = data[:,sigvarinds]
    C = C[sigvarinds,sigvarinds]

    n_observations,n_variables = size(X)
    X_min = minimum(X); X_max = maximum(X)
    m = sum(X,dims=1)/n_observations



    # now plot beautiful covariance

    if ax===nothing ax = plot(layout=(n_variables,n_variables), legend=false,
                              size=(200*n_variables,200*n_variables) )
    end
    
    hrange = collect(range(X_min,X_max,length=maximum([n_observations÷25,25])))

    for i in 1:n_variables
        for j in 1:i-1
            axs = ax[i,j]
            scatter!(axs, X[:,i], X[:,j], color=colors[1],markersize=0.4,alpha=0.9,
                     xlim=[X_min, X_max], ylim=[X_min, X_max] )
            annotate!(axs,  X_max*0.8, X_max*0.8,
                      text(  "$(Int(round(C[i,j]*100))/100)", [:gold,:lime][ Int(C[i,j]>=0)+1  ] )     )
            axs = ax[j,i]
            histogram2d!(axs, X[:,j], X[:,i], bins=hrange)
        end
    end
    
    for i in 1:n_variables
        for j in 1:n_variables
            axs = ax[i,j]
            if j>1 plot!(axs,yticks=nothing) end
            if i<n_variables plot!(axs,xticks=nothing) end
        end
    end



    for i in 1:n_variables
        axs = ax[i,i]
        histogram!(axs,X[:,i], bins=hrange,color=colors[2])
    end



    if varlabels !== nothing
        varlabels = varlabels[sigvarinds]   # reduce and reorder
        for i in 1:n_variables
            plot!(ax[i,1],           ylabel=varlabels[i])
            plot!(ax[n_variables,i], xlabel=varlabels[i])
        end
    end
    if title !== nothing plot!(ax[1,n_variables÷2+n_variables%2],title=title) end

    
    display(ax)
    return ax
end



# example usage
# myplot!(p::Plot, args...; kwargs...) = # do the thing
# myplot(args...; kwargs...) = myplot!(plot(), args...; kwargs...)

end