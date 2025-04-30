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

#include <CostFunctions/GenericCost.h>

float GenericCost::GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{
	// Here you should compute a scalar that indicates the "cost" of using the given velocity.
	return 0;
}

Vector2D GenericCost::GetGradient(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{
	// Here you should compute a 2D vector that is the gradient of the cost function at the given velocity.
	// If you do not know how to compute this, you can also leave out this method entirely, 
	// and then the parent class will automatically compute a numerical approximation of the gradient.
	// This is the same behavior as calling the parent version yourself:
	return CostFunction::GetGradient(velocity, agent, world);
}

void GenericCost::parseParameters(const CostFunctionParameters & params)
{
	// Here you may read custom parameters from "params", and call the parent version of parseParameters().
	// If you plan to do nothing except calling the parent version, you can also leave out this method entirely.
	CostFunction::parseParameters(params);
}
