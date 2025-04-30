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

#include <tools/Polygon2D.h>

#include <array>
#include <3rd-party/earcut/earcut.hpp>

Polygon2D::Polygon2D(const std::vector<Vector2D>& vertices) 
	: vertices_(vertices)
{
	// compute the list of edges
	const size_t n = vertices.size();
	edges_.resize(n);
	for (size_t i = 0; i < vertices.size(); ++i)
		edges_[i] = { vertices[i], vertices[(i + 1) % n] };

	// compute a triangulation
	computeTriangulation();
}

#pragma region [Triangulation]

std::vector<std::vector<Vector2D>> Polygon2D::GetTriangles() const
{
	std::vector<std::vector<Vector2D>> result;
	for (size_t i = 0; i < triangleIndices.size(); i += 3)
		result.push_back({ vertices_[triangleIndices[i]], vertices_[triangleIndices[i + 1]], vertices_[triangleIndices[i + 2]] });
	return result;
}

void Polygon2D::computeTriangulation()
{
	using Point = std::array<double, 2>;

	// Fill polygon structure with actual data. Any winding order works.
	std::vector<Point> boundary;
	for (const auto& pt : vertices_)
		boundary.push_back({ pt.x, pt.y });
	std::vector<std::vector<Point>> polygon = { boundary };

	// Run tessellation
	// Returns array of indices that refer to the vertices of the input polygon.
	// Three subsequent indices form a triangle. Output triangles are clockwise.
	triangleIndices = mapbox::earcut<size_t>(polygon);
}

#pragma endregion