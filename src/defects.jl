"""@docs

Generates random defects of the following types:
    - Pointwise
    - Linear
    - Ellipse

Will be extended to other polygons if FNO does not generalize.
"""

abstract type AbstractDefect end

struct Empty <: AbstractDefect end 

function boundary(nx,ny,::Empty)
    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 
    boundary
end 

struct Linear <: AbstractDefect
    nx
    ny
    start_coords
    end_coords
    orientation
    
    function Linear()
        new()
    end 

    function Linear(nx,ny)
        xcoord = rand(40:nx-40)
        ycoord = rand(40:ny-40)
        lenx = rand(1:(nx-40-40))
        leny = rand(1:(ny-40-40))
        ori = rand(-1:2:1)
        new(nx,ny,(xcoord,ycoord),(lenx,leny),ori)
    end 
end 

function boundary(nx,ny,::Linear)
    lin = Linear(nx,ny)
    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 
    
    xs = lin.start_coords[1]:(lin.start_coords[1] + lin.end_coords[1])
    ys = lin.start_coords[2]:(lin.start_coords[2] + lin.end_coords[2])
    
    ystart = ys[1]
    for i=1:length(xs)
        boundary[xs[i],ystart] = 1 
        ystart += lin.orientation
    end 

    boundary
end 

struct Point <: AbstractDefect end 

function boundary(nx,ny,::Point)

    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 

    xcoord = rand(10:nx-10)
    ycoord = rand(10:ny-10)
    
    boundary[xcoord,ycoord] = 1
    boundary
end 

struct Ellipse <: AbstractDefect
    xcenter
    ycenter
    radius_major
    radius_minor
    
    function Ellipse()
        new()
    end 

    function Ellipse(nx,ny)
        xc = rand(40:nx-40)
        yc = rand(40:ny-40)
        rmajor = rand(7:10)
        rminor = rand(1:5)
        new(xc,yc,rmajor,rminor)
    end 
end 

function boundary(nx,ny,::Ellipse)
    eli = Ellipse(nx,ny)
    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1

    xc,yc = eli.xcenter,eli.ycenter
    a,b = eli.radius_major,eli.radius_minor
    for i=1:nx
        for j=1:ny
            if sqrt(((i-xc)/a)^2 + ((j-yc)/b)^2) <= 1
                boundary[i,j] = 1
            end 
        end 
    end 
    boundary
end 

struct Spline <: AbstractDefect end 

function boundary(nx::Int,ny::Int,::Spline)
    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 

    sp = spline()
    spline2realdefect!(boundary,sp) 
    boundary
end 