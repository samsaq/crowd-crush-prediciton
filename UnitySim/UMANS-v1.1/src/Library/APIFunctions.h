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

#ifdef DLL_EXPORT
#ifdef WIN32
#define API_FUNCTION __declspec(dllexport) 
#else
#define API_FUNCTION 
#endif
#endif

extern "C"{
	/// <summary>A struct that describes the status of a single agent in the simulation.
	/// This struct is used for communication between the UMANS library and external applications.</summary>
	struct AgentData
	{
		/// The unique ID of the agent.
		int id;
		/// The x coordinate of the agent at the current time.
		float position_x;
		/// The y coordinate of the agent at the current time.
		float position_y;
		/// The x component of the agent's velocity at the current time.
		float velocity_x;
		/// The y component of the agent's velocity at the current time.
		float velocity_y;
		/// The x component of the agent's viewing direction at the current time.
		float viewingDirection_x;
		/// The y component of the agent's viewing direction at the current time.
		float viewingDirection_y;
	};

	/// <summary>Sets up a simulation based on a configuration file. 
	/// After this function call, the simulation will be ready for its first time step.</summary>
	/// <returns>true if the operation was successful; false otherwise, e.g. if the configuration file is invalid.</returns>
	API_FUNCTION  bool StartSimulation(const char* configFileName, int nrThreads);

	/// <summary>Gets the step size of the simulation, in seconds.</summary>
	/// <param ref="result_dt">[out] Will store the step size of the simulation.</param>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool GetSimulationTimeStep(float& result_dt);

	/// <summary>Sets the step size of the simulation, in seconds.</summary>
	/// <param ref="result_dt">The new simulation time step to use.</param>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool SetSimulationTimeStep(float dt);

	/// <summary>Performs the given number of simulation steps. 
	/// To obtain the resulting status of the simulation, use the GetAgentPositions() function.</summary>
	/// <param ref="nrSteps">The number of steps to perform.</param>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool DoSimulationSteps(int nrSteps);

	/// <summary>Gets the current status of all agents in the simulation.</summary>
	/// <param ref="result_agentData">[out] Will store a reference to an array of AgentData objects, 
	/// where each object describes the status of a single agent.</param>
	/// <param ref="result_nrAgents">[out] Will store the number of agents in the simulation, 
	/// i.e. the number of useful entries in the AgentData array.</param>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool GetAgentPositions(AgentData*& result_agentData, int& result_nrAgents);

	/// <summary>Overrides the position of some or all agents in the simulation. You can decide for yourself which agents are affected.</summary>
	/// <param ref="agentData">A reference to an array of AgentData objects, where each object describes the position of a single agent that you want to set.</param>
	/// <param ref="nrAgents">The number of entries in the AgentData array.</param>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool SetAgentPositions(AgentData* agentData, int nrAgents);

	/// <summary>Tries to add a new agent to the simulation.</summary>
	/// <param ref="x">The x-coordinate of the agent's starting position.</param>
	/// <param ref="y">The y-coordinate of the agent's starting position.</param>
	/// <param ref="radius">The agent's radius.</param>
	/// <param ref="prefSpeed">The agent's preferred speed.</param>
	/// <param ref="maxSpeed">The agent's maximum speed.</param>
	/// <param ref="maxAcceleration">The agent's maximum acceleration.</param>
	/// <param ref="policyID">The ID of the agent's policy to use for navigation.</param>
	/// <param ref="result_id">[out] Will store the ID of the agent that has been added.</param>
	/// <param ref="desiredID">[optional] A desired custom ID to use for the agent. This ID will be used if it is still available; 
	/// otherwise, the simulation will assign a different ID.</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the agent with the given ID does not exist.</returns>
	API_FUNCTION bool AddAgent(float x, float y, float radius, float prefSpeed, float maxSpeed, float maxAcceleration, int policyID, int& result_id, int desiredID = -1);

	/// <summary>Tries to remove a specific agent from the simulation.</summary>
	/// <param ref="id">The ID of the agent to remove.</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the agent with the given ID does not exist.</returns>
	API_FUNCTION bool RemoveAgent(int id);

	/// <summary>Changes the goal of a single agent in the simulation.</summary>
	/// <param ref="id">The ID of the agent to change.</param>
	/// <param ref="x">The x-coordinate of the agent's new goal.</param>
	/// <param ref="y">The y-coordinate of the agent's new goal.</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the agent with the given ID does not exist.</returns>
	API_FUNCTION bool SetAgentGoal(int id, float x, float y);

	/// <summary>Gets the visualization color of an agent.</summary>
	/// <param ref="id">The ID of the agent to check.</param>
	/// <param ref="r">[out] Will store the R component or the agent's new color, in the range [0-255].</param>
	/// <param ref="g">[out] Will store the G component or the agent's new color, in the range [0-255].</param>
	/// <param ref="b">[out] Will store the B component or the agent's new color, in the range [0-255].</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the agent with the given ID does not exist.</returns>
	API_FUNCTION bool GetAgentColor(int id, int& result_r, int& result_g, int& result_b);

	/// <summary>Changes the visualization color of an agent.</summary>
	/// <param ref="id">The ID of the agent to change.</param>
	/// <param ref="r">The R component or the agent's new color, in the range [0-255].</param>
	/// <param ref="g">The G component or the agent's new color, in the range [0-255].</param>
	/// <param ref="b">The B component or the agent's new color, in the range [0-255].</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the agent with the given ID does not exist.</returns>
	API_FUNCTION bool SetAgentColor(int id, int r, int g, int b);

	/// <summary>Cleans up some objects related to the simulation. Call this method just before you finish using the UMANS library.</summary>
	/// <returns>true if the operation was successful; false otherwise, i.e. if the simulation has not been initialized (correctly) yet.</returns>
	API_FUNCTION bool CleanUp();

	/// <summary>Tries to add a new obstacle polygon to the simulation.</summary>
	/// <param ref="points">the array of points (x1,y1,x2,y2...)</param>
	/// <param ref="nbr_points">The number of points (half size of array)</param>
	/// <returns>true if the operation was successful; false otherwise, 
	///  i.e. if the simulation has not been initialized (correctly) yet, or if the obstacle could not be created</returns>

	API_FUNCTION bool AddObstacle(float* points, int nbr_points);

}