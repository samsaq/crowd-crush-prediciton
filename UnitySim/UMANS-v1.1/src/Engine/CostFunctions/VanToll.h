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

#ifndef LIB_VANTOLL_H
#define LIB_VANTOLL_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>A custom navigation function by Wouter van Toll,
/// also used in the paper "Connecting global and local agent navigation via topology" (2019).</summary>
/// <remarks>
/// ### Notes: 
/// <list>
/// <item>This cost function already includes a goal-reaching component.</item>
/// <item>The <tt>%VanToll.xml</tt> policy in our example files reproduces the original algorithm.</item>
/// </list>
/// 
/// ### Name in XML files:
/// <tt>%VanToll</tt>
/// 
/// ### Parameters: 
/// <list>
/// <item>max_distance: The maximum distance-to-collision to consider.</item>
/// </list>
/// </remarks>
class VanToll : public CostFunction
{
private:
	float max_distance = 8;

public:
	VanToll() : CostFunction() {}
	virtual ~VanToll() {}
	const static std::string GetName() { return "VanToll"; }

	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const;
	void parseParameters(const CostFunctionParameters & params) override;
};

#endif //LIB_VANTOLL_H
