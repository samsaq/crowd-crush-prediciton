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

#ifndef LIB_WORLD_TORIC_H
#define LIB_WORLD_TORIC_H

#include <core/worldBase.h>

/// <summary>An implementation of WorldBase that treats the world as a torus, i.e. with horizontal and vertical wrap-around.</summary>
/// <remarks>The world can be interpreted as a rectangle with a certain width and height, 
/// where any agent that goes out of bounds re-appears on the other side. 
/// This wrap-around effect also affects nearest-neighbor queries.</remarks>
class WorldToric : public WorldBase
{
private:
	float width_;
	float height_;

public:
	/// <summary>Creates a WorldToric object with the given width and height.</summary>
	/// <param name="width">The width of the world, in meters.</param>
	/// <param name="height">The height of the world, in meters.</param>
	WorldToric(float width, float height);

	/// <summary>WorldToric's version of ComputeNeighbors().
	/// It performs multiple nearest-neighbor queries to account for the world's wrap-around effect.</summary>
	NeighborList ComputeNeighbors(const Vector2D& position, float search_radius, const Agent* queryingAgent) const override;
	
	/// <summary>WorldToric's version of DoStep_MoveAllAgents(). 
	/// It moves all agents forward and then possibly teleports them to the other end of the world, to simulate a wrap-around effect.</summary>
	virtual void DoStep_MoveAllAgents() override;

	/// <summary>Returns the width of this toric world.</summary>
	inline float GetWidth() const { return width_; }

	/// <summary>Returns the height of this toric world.</summary>
	inline float GetHeight() const { return height_; }

	/// <summary>Creates and returns two 2D vectors denoting this world's bounding box.</summary>
	/// <returns>A pair of Vector2D objects: the first denotes the bottom-left corner of the toric world, 
	/// and the second denotes the top-right corner.</returns>
	inline std::pair<Vector2D, Vector2D> GetBoundingBox() const
	{
		float halfW = width_ / 2.f, halfH = height_ / 2.f;
		return { Vector2D(-halfW,-halfH), Vector2D(halfW, halfH) };
	}

private:
	void computeNeighbors_Displaced(const Vector2D& position, const Vector2D& displacement, float search_radius, const Agent* queryingAgent,
		NeighborList& result) const;
};

#endif //LIB_WORLD_TORIC_H
