/* UMANS: Unified Microscopic Agent Navigation Simulator
** MIT License
** Copyright (C) 2018-2020  Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettré
**
** Permission is hereby granted, free of charge, to any person obtaining
** a copy of this software and associated documentation files (the
** "Software"), to deal in the Software without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Software, and to
** permit persons to whom the Software is furnished to do so, subject
** to the following conditions:
**
** The above copyright notice and this permission notice shall be
** included in all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
** OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
** NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
** LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
** ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**
** Contact: crowd_group@inria.fr
** Website: https://project.inria.fr/crowdscience/
** See the file AUTHORS.md for a list of all contributors.
*/

#ifndef LIB_POLYGON2D_H
#define LIB_POLYGON2D_H

#include <vector>
#include <tools/vector2D.h>

/// <summary>A wrapper for a 2D polygon defined by a list of Vector2D points.</summary>
/// <remarks>Upon creation, a Polygon2D object pre-computes a list of edges *and* a triangulation.</remarks>
class Polygon2D
{
private:
	std::vector<Vector2D> vertices_;
	std::vector<LineSegment2D> edges_;
	std::vector<size_t> triangleIndices;

public:
	/// <summary>Creates a Polygon2D with the given vertices.</summary>
	Polygon2D(const std::vector<Vector2D>& vertices);

	/// <summary>Returns the vertices of this Polygon2D.</summary>
	inline const std::vector<Vector2D>& GetVertices() const { return vertices_; }
	/// <summary>Returns the edges of this Polygon2D.</summary>
	inline const std::vector<LineSegment2D>& GetEdges() const { return edges_; }

	/// <summary>Creates and returns a triangulation of this Polygon2D. 
	/// The triangle indices have been precomputed; this method only converts these triangles to actual objects.</summary>
	std::vector<std::vector<Vector2D>> GetTriangles() const;

private:
	void computeTriangulation();

};
#endif //LIB_POLYGON2D_H
