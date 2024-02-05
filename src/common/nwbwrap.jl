__precompile__()
module NWBWrap



export pythontojulia_array, pja
export pythontojulia_eagerdata, pjed
export pythontojulia_dataframe, pjdf
export nwbobjecttodataframe, nwbdf
export timeseriestodataframe, tsdf
export timeintervalstodataframe, tidf

export PyObject, np, pd, nwb, hdmf


using PyCall
np = PyNULL()
pd = PyNULL()
nwb = PyNULL()
hdmf = PyNULL()


using DataFrames
pd = PyNULL()



function __init__()
    copy!(np, pyimport("numpy"))
    copy!(pd, pyimport("pandas"))
    copy!(nwb, pyimport("pynwb"))
    copy!(hdmf, pyimport("hdmf"))
end




#pyobject to array
function pythontojulia_array(pyarray::PyCall.PyObject)
    if py"type($(pyarray)[0])==str"
        c = convert(Array{String}, pyarray)
    elseif typeof(first(pyarray))<:Int
        convert(Array{Int64}, pyarray)
    elseif typeof(first(pyarray))<:AbstractFloat
        convert(Array{Float64}, pyarray)
    else
        c = pyarray
    end
end
pja = pythontojulia_array



#pyobject hdmf dataset to array
function pythontojulia_eagerdata(pydata::PyCall.PyObject)
    if py"hasattr($pydata,'data')"
        py"$(pydata).data[:]"
    else
        error("pythontojulia_eagerdata: pydata does not have a data field")
    end
end
pjed = pythontojulia_eagerdata





# pyobject to dataframe
function pythontojulia_dataframe(df_pd::PyCall.PyObject)
    df = DataFrame()
    for col in df_pd.columns
        v = getproperty(df_pd, col).values
        if py"type($v[0])==str"             # convert strings to symbols, every other type should work as is
            v = convert.(String, v)
        end
        df[!, col] = v
    end
    df
end
pjdf = pythontojulia_dataframe




function nwbobjecttodataframe(nwbobject::PyCall.PyObject)
    pythontojulia_dataframe(nwbobject.to_dataframe())
end
nwbdf = nwbobjecttodataframe





function timeseriestodataframe(timeseriesobject::PyCall.PyObject)
    tso = DataFrame( [  py"""$(timeseriesobject).timestamps[:]""",
                       py"""$(timeseriesobject).data[:]"""],
                       [:timestamps, :data] )
    return tso
end
function timeseriestodataframe(timeseriesobject::PyCall.PyObject,start::Float64,stop::Float64)
    tso = DataFrame( [  py"""$(timeseriesobject).timestamps[:]""",
                       py"""$(timeseriesobject).data[:]"""],
                       [:timestamps, :data] )
    mask = (tso[!,:timestamps].>=start) .& (tso[!,:timestamps].<=stop)
    return tso[mask,:]
end
tsdf = timeseriestodataframe



function timeintervalstodataframe(timeintervalsobject::PyCall.PyObject)
    DataFrame( [ pja(c) for c in timeintervalsobject.columns], collect(timeintervalsobject.colnames) )
end
function timeintervalstodataframe(timeintervalsobject::PyCall.PyObject, start::Float64,stop::Float64; strict=false)
    tso = DataFrame( [ pja(c) for c in timeintervalsobject.columns], collect(timeintervalsobject.colnames) )
    if strict
        mask = (tso[!,:start_time].>=start) .& (tso[!,:stop_time].<=stop)
    else 
        mask = (tso[!,:stop_time].>=start) .& (tso[!,:start_time].<=stop)
    end
    return tso[mask,:]
end
tidf = timeintervalstodataframe



end
