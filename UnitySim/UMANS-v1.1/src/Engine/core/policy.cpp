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

#include <core/policy.h>
#include <core/agent.h>
#include <core/worldBase.h>

Policy::~Policy()
{
	// delete all cost functions
	for (auto& costFunction : cost_functions_)
		delete costFunction.first;
	cost_functions_.clear();
}

float Policy::getInteractionRange() const
{
	float range = 0;
	for (const auto& costFunction : cost_functions_)
		range = std::max(range, costFunction.first->GetRange());

	return range;
}

Vector2D Policy::ComputeAcceleration(Agent* agent, WorldBase * world)
{
	const Vector2D& currentVelocity = agent->getVelocity();
	float dt = world->GetDeltaTime();

	// if the agent does not want to move, return a deceleration vector
	if (agent->getPreferredVelocity().isZero())
		return -currentVelocity / dt;

	// compute a new acceleration using a certain optimization method

	// a) Gradient following: compute acceleration from cost-function gradients
	if (optimizationMethod_ == OptimizationMethod::GRADIENT)
		return getAccelerationFromGradient(agent, world);

	// b) Global optimization or sampling
	
	// - compute the ideal velocity according to the cost functions
	const Vector2D& bestVelocity = (optimizationMethod_ == OptimizationMethod::GLOBAL
			? getBestVelocityGlobal(agent, world)
			: getBestVelocitySampling(agent, world, samplingParameters_));

	// - convert this to an acceleration using a relaxation time
	//   Note: the relaxation time should be at least the length of a frame.
	return (bestVelocity - currentVelocity) / std::max(relaxationTime_, dt);
}

float Policy::ComputeCostForVelocity(const Vector2D& velocity, Agent* agent, WorldBase* world)
{
	// compute the cost for this velocity
	float totalCost = 0;
	for (auto& costFunction : cost_functions_)
		totalCost += costFunction.second * costFunction.first->GetCost(velocity, agent, world);

	return totalCost;
}

Vector2D Policy::getAccelerationFromGradient(Agent* agent, WorldBase * world)
{
	const Vector2D& CurrentVelocity = agent->getVelocity();

	// sum up the gradient of all cost functions
	Vector2D TotalGradient(0, 0);
	for (auto& costFunction : cost_functions_)
		TotalGradient += costFunction.second * costFunction.first->GetGradientFromCurrentVelocity(agent, world);

	// move in the opposite direction of this gradient
	return -1 * TotalGradient;
}

Vector2D Policy::getBestVelocityGlobal(Agent* agent, WorldBase * world)
{
	// Note: True global optimization only works if this policy has a single cost function, 
	// because it requires a closed-form solution that has to be implemented per function.
	// If no such closed-form solution is given (or if there are multiple cost functions), we have to resort to sampling.

	return cost_functions_.size() == 1
		? cost_functions_[0].first->GetGlobalMinimum(agent, world)
		: getBestVelocitySampling(agent, world, SamplingParameters::ApproximateGlobalOptimization());
}

Vector2D Policy::getBestVelocitySampling(Agent* agent, WorldBase * world, const SamplingParameters& params)
{
	return CostFunction::ApproximateGlobalMinimumBySampling(agent, world, params, cost_functions_);
}

Vector2D Policy::ComputeContactForces(Agent* agent, WorldBase * world)
{
	Vector2D totalForce(0, 0);
	const Vector2D& position = agent->getPosition();
	const float radius = agent->getRadius();

	if (contactForceScale_ > 0)
	{
		const auto& neighbors = agent->getNeighbors();

		// check all colliding agents
		for (const auto& neighborAgent : neighbors.first)
		{
			const auto& diff = position - neighborAgent.GetPosition();
			const float intersectionDistance = radius + neighborAgent.realAgent->getRadius() - diff.magnitude();
			if (intersectionDistance > 0)
				totalForce += diff.getnormalized() * (contactForceScale_ * intersectionDistance);
		}

		// check all colliding obstacles
		for (const auto& neighborObstacle : neighbors.second)
		{
			const auto& nearest = nearestPointOnLine(position, neighborObstacle.first, neighborObstacle.second, true);
			const auto& diff = position - nearest;
			const float intersectionDistance = radius - diff.magnitude();
			if (intersectionDistance > 0)
			{
				const auto& F = diff.getnormalized() * (contactForceScale_ * intersectionDistance);
				totalForce = totalForce + F;
			}
		}
	}

	return totalForce;
}

void Policy::AddCostFunction(CostFunction* costFunction, const CostFunctionParameters &params)
{
	float coefficient = 1;
	params.ReadFloat("coeff", coefficient);
	costFunction->parseParameters(params);
	cost_functions_.push_back({ costFunction, coefficient });
}

bool Policy::OptimizationMethodFromString(const std::string &method, Policy::OptimizationMethod& result)
{
	if (method == "gradient")
		result = OptimizationMethod::GRADIENT;
	else if (method == "sampling")
		result = OptimizationMethod::SAMPLING;
	else if (method == "global")
		result = OptimizationMethod::GLOBAL;
	else
		return false;
	return true;
}

bool SamplingParameters::BaseFromString(const std::string &method, SamplingParameters::Base& result)
{
	if (method == "zero")
		result = Base::ZERO;
	else if (method == "current velocity")
		result = Base::CURRENT_VELOCITY;
	else
		return false;
	return true;
}

bool SamplingParameters::BaseDirectionFromString(const std::string &method, SamplingParameters::BaseDirection& result)
{
	if (method == "unit")
		result = BaseDirection::UNIT;
	else if (method == "current velocity")
		result = BaseDirection::CURRENT_VELOCITY;
	else if (method == "preferred velocity")
		result = BaseDirection::PREFERRED_VELOCITY;
	else
		return false;
	return true;
}

bool SamplingParameters::TypeFromString(const std::string &method, SamplingParameters::Type& result)
{
	if (method == "regular")
		result = Type::REGULAR;
	else if (method == "random")
		result = Type::RANDOM;
	else
		return false;
	return true;
}

bool SamplingParameters::RadiusFromString(const std::string &method, SamplingParameters::Radius& result)
{
	if (method == "preferred speed")
		result = Radius::PREFERRED_SPEED;
	else if (method == "maximum speed")
		result = Radius::MAXIMUM_SPEED;
	else if (method == "maximum acceleration")
		result = Radius::MAXIMUM_ACCELERATION;
	else
		return false;
	return true;
}