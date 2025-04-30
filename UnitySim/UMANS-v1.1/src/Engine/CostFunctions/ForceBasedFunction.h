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

#ifndef LIB_FORCE_BASED_FUNCTION_H
#define LIB_FORCE_BASED_FUNCTION_H

#include <core/costFunction.h>

/// <summary>An abstract class for a cost function that is based on forces.</summary>
/// <remarks>This class implements GetCost() and GetGradient() in a generic way.
/// In return, it introduces the abstract method ComputeForce() which should compute a 2D force vector. 
/// 
/// If you want to implement a force-based navigation function, consider inheriting from this class:
/// you can then focus on just computing the force itself, and everything else will be handled for you.</remarks>
class ForceBasedFunction : public CostFunction
{
protected:
	ForceBasedFunction() : CostFunction() {}
	virtual ~ForceBasedFunction() {}

public:
	/// <summary>Computes the cost of a given velocity, in a way that is specific for force-based functions.</summary>
	/// <remarks>In the case of ForceBasedFunction, the cost depends on the difference to the target velocity 
	/// (the velocity that the agent would reach by using the method's force).</remarks>
	/// <param name="velocity">The velocity for which the cost is requested; ForceBasedFunction actually does not use this.</param>
	/// <param name="agent">The agent that would use the requested velocity.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>A floating-point cost, derived from the result of ComputeForce().</param>
	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	/// <summary>Computes the gradient of the cost, in a way that is specific for force-based functions.</summary>
	/// <remarks>In the case of ForceBasedFunction, the gradient points towards the target velocity 
	/// (the velocity that the agent would reach by using the method's force), with a magnitude that will always lead to this target velocity.</remarks>
	/// <param name="velocity">The velocity for which the cost is requested; ForceBasedFunction actually does not use this.</param>
	/// <param name="agent">The agent that would use the requested velocity.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>The gradient of the cost function, derived from the result of ComputeForce().</param>
	virtual Vector2D GetGradient(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	/// <summary>Computes the gradient of the cost at the agent's current velocity, in a way that is specific for force-based functions.</summary>
	/// <remarks>In the case of ForceBasedFunction, this should simply return -F/m, and we can skip some computations from the regular GetGradient() function.</remarks>
	/// <param name="agent">An agent in the simulation.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>The gradient of the cost function, derived from the result of ComputeForce().</param>
	virtual Vector2D GetGradientFromCurrentVelocity(Agent* agent, const WorldBase * world) const override;

	/// <summary>Computes the globally optimal velocity, in a way that is specific for force-based functions.</summary>
	/// <remarks>In the case of ForceBasedFunction, this is the velocity that the agent would reach if it uses the force for one time step.</remarks>
	/// <param name="agent">The agent that would use the requested velocity.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>The velocity resulting from using this cost function's force.</param>
	virtual Vector2D GetGlobalMinimum(Agent* agent, const WorldBase * world) const override;
	
	/// <summary>Parses the parameters of the cost function.</summary>
	/// <remarks>By default, ForceBasedFunction simply calls the parent version, i.e. CostFunction::parseParameters().
	/// Override this method to let it load any additional parameters of your interest.</remarks>
	/// <param name="params">An XML block that contains all parameters of this cost function.</param>
	void parseParameters(const CostFunctionParameters & params) override;

protected:
	/// <summary>Computes and returns a 2D force vector that the agent experiences.</summary>
	/// <remarks>Subclasses of ForceBasedFunction should implement this method.</remarks>
	/// <param name="agent">The agent for which a force is requested.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>A 2D vector indicating the force that the agent experiences according to a specific model.</returns>
	virtual Vector2D ComputeForce(Agent* agent, const WorldBase* world) const = 0;
	
private:
	/// <summary>Computes the velocity that the agent would reach if it uses the result of ComputeForce() for one timestep.</summary>
	/// <param name="agent">The agent that would use the requested velocity.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>The velocity resulting from using this cost function's force.</param>
	Vector2D ComputeTargetVelocity(Agent* agent, const WorldBase* world) const;
};

#endif //LIB_FORCE_BASED_FUNCTION_H
