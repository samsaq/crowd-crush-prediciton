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

#include "UMANSOpenGLWidget.h"
#include "UMANSQtGuiApplication.h"

#include <QMouseEvent>
#include <QTimer>
#include <QOpenGLShaderProgram>
#include <QCoreApplication>
#include <QFileDialog>
#include <QMessageBox>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <tools/Color.h>
#include <tools/HelperFunctions.h>

#include <Engine/core/worldToric.h>
#include <Engine/core/policy.h>

// Vertex shader: calculates the position of a vertex in the OpenGL view, and passes a color onto the fragment shader.
static const char *vertexShaderSource =
"attribute vec3 vertex;\n"
"attribute vec3 color;\n"
"varying vec3 f_color;\n"
"uniform mat4 projectionMatrix;\n"
"void main() {\n"
"   f_color = color;\n"
"   gl_Position = projectionMatrix * vec4(vertex.x, vertex.y, vertex.z, 1.0);\n"
"}\n";

// Fragment shader: sets the color.
static const char *fragmentShaderSource =
"varying vec3 f_color;\n"
"void main(void) {\n"
"	gl_FragColor = vec4(f_color.r, f_color.g, f_color.b, 1.0);\n"
"}\n";

static const std::string Target_Grid = "grid";
static const std::string Target_Environment_Solid = "env1";
static const std::string Target_Environment_Contours = "env2";
static const std::string Target_Agents_Solid = "ag1";
static const std::string Target_Agents_Contours = "ag2";

static const Qt::Key Key_SelectAgent = Qt::Key::Key_Shift;
static const Qt::Key Key_PauseResume = Qt::Key::Key_Space;
static const Qt::Key Key_DeleteAgent = Qt::Key::Key_Delete;
static const Qt::Key Key_DeleteRoute = Qt::Key::Key_R;

static const int nrThreadsForSimulation = 4;

static const std::string DefaultScenarioFile = "../examples/Circle15-RVO.xml";

const double Depth_Grid = 0.1;
const double Depth_GridBoundary = 0.15;
const double Depth_Obstacles = 0.2;
const double Depth_Agents = 0.5;

UMANSOpenGLWidget::UMANSOpenGLWidget(QWidget *parent) : QOpenGLWidget(parent), simulationTimer(nullptr), simulator(nullptr), activeAgent(nullptr)
{
	panning = false;
	simulationRunning = false;
	inAgentSelectMode = false;

	writeCSVOutput = false;
	makeScreenshotsPerFrame = false;

	showCostFunction = false;

	setFocusPolicy(Qt::FocusPolicy::ClickFocus);

	mainApplication = (UMANSQtGuiApplication*)parent;
}

void UMANSOpenGLWidget::resetVisualization(bool deleteCurrentVisualization)
{
	if (deleteCurrentVisualization)
	{
		// delete all visualization
		for (auto& vis : visualizationData)
			vis.second->ClearData();
		ZoomToFit();
	}

	drawEnvironment();
	drawSimulation();

	// visualize the new simulation time
	updateSimulationTimerText();

	update();
}

void UMANSOpenGLWidget::showErrorMessage(const std::string& title, const std::string& details) const
{
	QMessageBox messageBox(mainApplication);
	messageBox.critical(0, QString::fromStdString(title), QString::fromStdString(details));
}

void UMANSOpenGLWidget::startNewSimulation(const std::string& scenarioFilename)
{
	bool firstTime = simulator == nullptr;

	// try to load a new simulation
	CrowdSimulator* newSimulator = CrowdSimulator::FromConfigFile(scenarioFilename);
	if (newSimulator != nullptr)
	{
		// delete the current simulation, if applicable
		if (simulator != nullptr)
			delete simulator;

		setActiveAgent(nullptr);

		simulator = newSimulator;

		// Prepare a timer for update events, using the current value of playbackMultiplier.
		// This works for the very first simulation, because playbackMultiplier=1 by default.
		// In other cases, the new simulation may use a different framerate than the previous one, so we need to re-initialize the timer.
		SetPlaybackMultiplier(playbackMultiplier);
	}
	else
	{
		showErrorMessage("Error",
			"Failed to start a new simulation based on the file \"" + scenarioFilename + "\". Please see the console for more information.");
	}

	// reset the UI
	resetVisualization(!firstTime);
}

UMANSOpenGLWidget::~UMANSOpenGLWidget()
{
	for (auto vis : visualizationData)
		delete vis.second;
	visualizationData.clear();
	delete simulationTimer;

	delete simulator;
}

#pragma region [Slots]

void UMANSOpenGLWidget::PlaySimulation()
{
	simulationRunning = true;
	simulationTimer->start();

	mainApplication->GetUI().Button_Play->setEnabled(false);
	mainApplication->GetUI().Button_Pause->setEnabled(true);
}

void UMANSOpenGLWidget::PauseSimulation()
{
	simulationRunning = false;
	simulationTimer->stop();

	mainApplication->GetUI().Button_Play->setEnabled(true);
	mainApplication->GetUI().Button_Pause->setEnabled(false);
}

void UMANSOpenGLWidget::ResetSimulation()
{
	startNewSimulation(simulator->GetScenarioFilename());
	PauseSimulation();
}

void UMANSOpenGLWidget::ToggleScreenshots()
{
	makeScreenshotsPerFrame = !makeScreenshotsPerFrame;
}

void UMANSOpenGLWidget::ToggleGrid()
{
	// show or hide the grid
	visualizationData[Target_Grid]->ToggleEnabled();
	update();
}

void UMANSOpenGLWidget::ToggleShowCostFunction()
{
	// show or hide the cost function of the selected agent
	showCostFunction = !showCostFunction;
	update();
}

void UMANSOpenGLWidget::SetPlaybackMultiplier(int value)
{
	playbackMultiplier = value;

	// prepare a timer for automatically running the simulation
	if (simulationTimer == nullptr)
		simulationTimer = new QTimer(this);
	else
	{
		disconnect(simulationTimer, SIGNAL(timeout()), this, SLOT(updateSimulation()));
		if (simulationTimer->isActive())
			simulationTimer->stop();
	}

	// determine how many milliseconds should be between two simulation updates
	int desiredRefreshTime = (int)(1000 * simulator->GetWorld()->GetDeltaTime() / playbackMultiplier);

	// schedule a Qt timer for periodically updating the simulation
	simulationTimer->setInterval(desiredRefreshTime);
	connect(simulationTimer, SIGNAL(timeout()), this, SLOT(updateSimulation()));

	// if the simulation was running before we changed this multiplier, make sure it continues running
	if (simulationRunning)
		simulationTimer->start();
}

void UMANSOpenGLWidget::OpenScenarioFileDialog()
{
	PauseSimulation();
	// get the folder that contained the current environment
	const std::string& currentEnvFilename = simulator->GetScenarioFilename();
	const std::string& currentEnvFolder = currentEnvFilename.substr(0, currentEnvFilename.find_last_of('/'));

	// open a dialog that starts at that folder
	const QString& fileName = QFileDialog::getOpenFileName(this,
		tr("Open Scenario File..."), QString::fromStdString(currentEnvFolder),
		tr("Scenarios (*.xml);;All Files (*)"));

	if (fileName.isEmpty())
		return;

	// reset the simulation
	startNewSimulation(fileName.toStdString());
}

void UMANSOpenGLWidget::ZoomToFit()
{
	// find the bbox and aspect ratio of the environment
	std::pair<Vector2D, Vector2D> bbox;
	if (simulator->GetWorld()->GetType() == WorldBase::Type::TORIC_WORLD)
		bbox = dynamic_cast<const WorldToric*>(simulator->GetWorld())->GetBoundingBox();
	else
		bbox = { {-10, -10}, {10, 10} };

	double width_environment = bbox.second.x - bbox.first.x;
	double height_environment = bbox.second.y - bbox.first.y;
	double aspect_environment = width_environment / height_environment;

	// compare it with the aspect ratio of the window
	double width_window = windowSize.width();
	double height_window = windowSize.height();
	double aspect_window = width_window / height_window;

	double xmin = bbox.first.x, ymin = bbox.first.y, xmax = bbox.second.x, ymax = bbox.second.y;

	// if the environment is relatively wide, show extra world space vertically
	if (aspect_environment > aspect_window)
	{
		double viewportHeight = width_environment / aspect_window;
		double extraY = (viewportHeight - height_environment) / 2.0;
		ymin -= extraY;
		ymax += extraY;
	}

	// if the environment is relatively high, show extra world space horizontally
	else
	{
		double viewportWidth = height_environment * aspect_window;
		double extraX = (viewportWidth - width_environment) / 2.0;
		xmin -= extraX;
		xmax += extraX;
	}

	// store the new view bounds, and refresh the screen
	viewBounds = QRectF(QPointF(xmin, ymax), QPointF(xmax, ymin));
	update();
}

std::string getScenarioNameFromFullPath(const std::string& scenarioPath)
{
	// remove the path path
	auto lastSlash = scenarioPath.find_last_of("/");
	std::string scenarioName = (lastSlash == std::string::npos ? scenarioPath : scenarioPath.substr(lastSlash + 1));

	// remove the file extension
	auto lastPoint = scenarioName.find_last_of(".");
	scenarioName = (lastPoint == std::string::npos ? scenarioName : scenarioName.substr(0, lastPoint));

	return scenarioName;
}

void UMANSOpenGLWidget::ToggleCSVOutput()
{
	writeCSVOutput = !writeCSVOutput;
	if (writeCSVOutput)
	{
		const auto& scenarioName = getScenarioNameFromFullPath(simulator->GetScenarioFilename());
		simulator->StartCSVOutput("../output/" + scenarioName + "/", true); // true = write output continuously
	}
	else
		simulator->StopCSVOutput();
}

void UMANSOpenGLWidget::MakeScreenshot()
{
	const QRect rect(0, 0, width(), height());
	QPixmap pixmap = grab(rect);

	const std::string& scenarioName = getScenarioNameFromFullPath(simulator->GetScenarioFilename());
	const std::string& dirName = "../screenshots/" + scenarioName + "/";
	bool dirExists = HelperFunctions::CreateDirectoryIfNonExistent(dirName);

	if (dirExists)
	{
		std::string filename = dirName + HelperFunctions::ToStringWithLeadingZeros((int)(simulator->GetWorld()->GetCurrentTime() * 10), 4) + ".png";
		pixmap.save(filename.c_str());
	}
}

#pragma endregion

#pragma region [Visualization helpers]

QVector3D pointToQVector3D(const Vector2D& pt, const double depth)
{
	return QVector3D(pt.x, pt.y, depth);
}

QSizeF UMANSOpenGLWidget::getScreenToWorldScale() const
{
	return QSizeF(
		viewBounds.size().width() / windowSize.width(),
		viewBounds.size().height() / windowSize.height()
	);
}

Vector2D UMANSOpenGLWidget::screenToWorld(const QPoint& screenPoint) const
{
	float xx = (float)screenPoint.x() / (float)windowSize.width() * viewBounds.size().width() + viewBounds.left();
	float yy = (float)screenPoint.y() / (float)windowSize.height()* viewBounds.size().height() + viewBounds.top();

	return Vector2D(xx, yy);
}

void UMANSOpenGLWidget::addSegmentsToBuffer(const std::vector<LineSegment2D>& segments, const QColor& color, const std::string& target, const double depth)
{
	auto vis = visualizationData[target];

	for (const auto& segment : segments)
	{
		vis->AddData(pointToQVector3D(segment.first, depth), color);
		vis->AddData(pointToQVector3D(segment.second, depth), color);
	}
}

void UMANSOpenGLWidget::addContourToBuffer(const std::vector<Vector2D>& points, const QColor& color, const std::string& target, const double depth)
{
	auto vis = visualizationData[target];

	for (size_t i = 0; i < points.size(); ++i)
	{
		vis->AddData(pointToQVector3D(points[i], depth), color);
		vis->AddData(pointToQVector3D(points[(i + 1) % points.size()], depth), color);
	}
}

void UMANSOpenGLWidget::addPointsToBuffer(const std::vector<Vector2D>& points, const QColor& color, const std::string& target, const double depth)
{
	auto vis = visualizationData[target];

	for (const auto& pt : points)
		vis->AddData(pointToQVector3D(pt, depth), color);
}

#pragma endregion

#pragma region [Drawing the simulation components]

void UMANSOpenGLWidget::drawSimulation()
{
	visualizationData[Target_Agents_Solid]->ClearData();
	visualizationData[Target_Agents_Contours]->ClearData();

	for (const auto& agent : simulator->GetWorld()->GetAgents())
		drawAgent(*agent);
}

void UMANSOpenGLWidget::drawAgent(Agent& agent)
{
	const bool isActiveAgent = activeAgent != nullptr && agent.getID() == activeAgent->getID();

	// draw an arrow on top of the disk
	const float radius = agent.getRadius();
	const auto& dir = agent.getViewingDirection() * (0.65*radius);
	const auto& trans = Vector2D(-dir.y, dir.x);

	const auto& triangleBase = agent.getPosition();

	addPointsToBuffer({
		triangleBase - trans,
		triangleBase + trans,
		triangleBase + dir
		},
		QColor(0, 0, 0),
		Target_Agents_Solid,
		Depth_Agents);

	// draw the disk
	const Color& agentColor = agent.getColor();
	QColor agentQColor(agentColor.r, agentColor.g, agentColor.b);
	addPointsToBuffer(approximateDisk_Triangles(agent.getPosition(), radius), agentQColor, Target_Agents_Solid, Depth_Agents);

	if (isActiveAgent)
	{
		// highlight the agent
		addPointsToBuffer(approximateDisk_Triangles(agent.getPosition(), radius*1.25), QColor(0, 0, 0), Target_Agents_Solid, Depth_Agents);

		// draw the goal
		addPointsToBuffer(approximateDisk_Triangles(agent.getGoal(), radius, 8), QColor(0, 255, 128), Target_Agents_Solid, Depth_Agents);

		// draw the values of the agent's cost function
		if (showCostFunction)
			drawAgentCostCircle(agent);
	}

	/*// draw the trajectory that the agent has traversed so far
	QColor agentQColorLight(255 - (255 - agentColor.r) / 3, 255 - (255 - agentColor.g) / 3, 255 - (255 - agentColor.b) / 3);

	for (const auto& pt : agent.GetTrajectory())
		addPointsToBuffer(approximateDisk_Triangles(pt, radius / 3.0), agentQColorLight, Target_DynamicData_Solid, Depth_Obstacles);*/
}

QColor colorFromCost(float cost, float minCost, float maxCost)
{
	// convert the cost to a fraction in [0,1] between min and max
	const float minCost2 = std::max(minCost, -100.f); //0
	const float maxCost2 = std::min(maxCost, 100.f); //50
	const float cost2 = std::min(std::max(cost, minCost2), maxCost2);
	float frac = (cost2 - minCost2) / (maxCost2 - minCost2);
	frac = HelperFunctions::Clamp(frac, 0.f, 1.f);
	if (isnan(cost) || minCost2 >= maxCost2)
		frac = 1;

	// convert this to a HSV representation: from blue to purple
	const int minH = 360 + 250; const int maxH = 300;
	int H = (int)(minH + frac * (maxH - minH)) % 360;

	return QColor::fromHsv(H, 255, 255);
}

void UMANSOpenGLWidget::drawAgentCostCircle_AddPoint(const Vector2D& basePos, const std::pair<Vector2D, float>& data, const float minCost, const float maxCost)
{
	Vector2D v(basePos + data.first);
	QVector3D vec3D(v.x, v.y, (float)(Depth_Agents - 0.01));
	visualizationData[Target_Agents_Solid]->AddData(vec3D, colorFromCost(data.second, minCost, maxCost));
}

void UMANSOpenGLWidget::drawAgentCostCircle(Agent& agent)
{
	// - collect cost values for a set of sample velocities

	WorldBase* world = simulator->GetWorld();
	const float maxSpeed = agent.getMaximumSpeed();
	const float ds = 0.025f;
	const int dA = 1;
	std::vector<std::vector<std::pair<Vector2D, float>>> costs;

	Vector2D zeroVector(0, 0);
	const float zeroVectorCost = agent.getPolicy()->ComputeCostForVelocity(zeroVector, &agent, world);
	float minCost = zeroVectorCost;
	Vector2D minCostVelocity(0, 0);
	float maxCost = zeroVectorCost;
	costs.push_back({ {zeroVector, zeroVectorCost } });

	for (float s = ds; s < maxSpeed; s += ds)
	{
		Vector2D base(s, 0);
		std::vector<std::pair<Vector2D, float>> costsForSpeed;
		for (int A = 0; A < 360; A += dA)
		{
			float angleRadians = (float)(A / 180.0 * PI);
			const Vector2D& v = rotateCounterClockwise(base, angleRadians);
			float cost = agent.getPolicy()->ComputeCostForVelocity(v, &agent, world);
			minCost = std::min(cost, minCost);
			if (cost == minCost)
				minCostVelocity = v;
			maxCost = std::max(cost, maxCost);
			costsForSpeed.push_back({ v, cost });
		}
		costs.push_back(costsForSpeed);
	}

	// If the agent uses ORCA, then the code above has caused its internal ORCA data to be overwritten. 
	// To prevent changes in behavior, we now "invalidate" this internal data, so that it will be recomputed in the next frame as usual.
	agent.GetOrcaSolution().currentSimulationTime = -1;

	// - draw these values in a circle

	const Vector2D& agentPos = agent.getPosition();

	for (size_t i = 1; i < costs.size(); ++i)
	{
		const auto& currentList = costs[i];
		const auto& prevList = costs[i - 1];

		for (size_t j = 0; j < currentList.size(); ++j)
		{
			const auto& data1 = currentList[j];
			const auto& data2 = currentList[(j + 1) % currentList.size()];

			if (i == 1)
			{
				// 1 triangle
				const auto& data3 = prevList[0];

				drawAgentCostCircle_AddPoint(agentPos, data1, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data2, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data3, minCost, maxCost);
			}
			else
			{
				// 2 triangles
				const auto& data3 = prevList[(j + 1) % prevList.size()];
				const auto& data4 = prevList[j];

				drawAgentCostCircle_AddPoint(agentPos, data1, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data2, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data3, minCost, maxCost);

				drawAgentCostCircle_AddPoint(agentPos, data1, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data3, minCost, maxCost);
				drawAgentCostCircle_AddPoint(agentPos, data4, minCost, maxCost);
			}

		}
	}

	// - highlight certain velocities

	addContourToBuffer(
		approximateDisk_Outline(agentPos + agent.getPreferredVelocity(), 0.1f),
		QColor(128, 128, 128), Target_Agents_Contours, Depth_Agents + 0.2);

	addContourToBuffer(
		approximateDisk_Outline(agentPos + minCostVelocity, 0.1f),
		QColor(255, 255, 255), Target_Agents_Contours, Depth_Agents + 0.2);

	// - draw ORCA lines, if they exist

	const auto& orcaSolution = agent.GetOrcaSolution();
	const QColor& orcaColor = orcaSolution.isFeasible ? QColor(0, 0, 0) : QColor(255, 0, 0);
	for (const auto& line : orcaSolution.orcaLines)
	{
		Vector2D dir = line.direction.getnormalized();
		Vector2D drawLinePoint = agentPos + line.point;
		addSegmentsToBuffer({ LineSegment2D(drawLinePoint - 2 * dir, drawLinePoint + 2 * dir) }, orcaColor, Target_Agents_Contours, Depth_Agents + 0.2);
	}
}

void UMANSOpenGLWidget::drawEnvironment(const bool refresh)
{
	const WorldBase* world = simulator->GetWorld();

	// clear the old visualization?
	if (refresh)
	{
		visualizationData[Target_Environment_Solid]->ClearData();
		visualizationData[Target_Environment_Contours]->ClearData();
	}

	// - draw the interior of all obstacles
	for (const auto& ob : world->GetObstacles())
		for (const auto& t : ob.GetTriangles())
			addPointsToBuffer(t, QColor(195, 195, 195), Target_Environment_Solid, Depth_Obstacles);

	// - draw a grid
	drawGrid(refresh);
}

void UMANSOpenGLWidget::drawGrid(bool refresh)
{
	const WorldBase* world = simulator->GetWorld();

	// clear the old visualization?
	if (refresh)
	{
		visualizationData[Target_Grid]->ClearData();
	}
	
	// Determine the bounds of the grid
	std::pair<Vector2D, Vector2D> bbox;
	// - in toric worlds, use the actual width and height of the world
	if (simulator->GetWorld()->GetType() == WorldBase::TORIC_WORLD)
	{
		bbox = dynamic_cast<const WorldToric*>(world)->GetBoundingBox();

		// draw the bounding box in a different color
		std::vector<Vector2D> grid_boundary = {
			{bbox.first.x,  bbox.first.y },
			{bbox.first.x,  bbox.second.y},
			{bbox.second.x, bbox.second.y},
			{bbox.second.x, bbox.first.y }
		};

		addContourToBuffer(grid_boundary, QColor(255, 0, 0), Target_Grid, Depth_GridBoundary);
	}
	// - otherwise, use a default range of -100 to +100
	else
		bbox = { {-100,-100}, {100, 100} };

	std::vector<LineSegment2D> grid, grid_major;
	const int majorSteps = 10;

	// vertical lines
	for (int x = (int)ceil(bbox.first.x); x <= bbox.second.x; ++x)
	{
		LineSegment2D seg(Vector2D(x, bbox.first.y), Vector2D(x, bbox.second.y));
		if (x % majorSteps == 0)
			grid_major.push_back(seg);
		else
			grid.push_back(seg);
	}

	// horizontal lines
	for (int y = (int)ceil(bbox.first.y); y <= bbox.second.y; ++y)
	{
		LineSegment2D seg(Vector2D(bbox.first.x, y), Vector2D(bbox.second.x, y));
		if (y % majorSteps == 0)
			grid_major.push_back(seg);
		else
			grid.push_back(seg);
	}

	addSegmentsToBuffer(grid_major, QColor(120, 120, 120), Target_Grid, Depth_Grid);
	addSegmentsToBuffer(grid, QColor(220, 220, 220), Target_Grid, Depth_Grid);
}

#pragma endregion

#pragma region [Main OpenGL functions]

void UMANSOpenGLWidget::initializeGL()
{
	f = QOpenGLContext::currentContext()->functions();
	f->initializeOpenGLFunctions();

	// set the background color
	glClearColor(1, 1, 1, 1);

	// Add the vertex and fragment shaders to the program.
	program.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
	program.addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);

	// Tell the shader program that the "vertex" and "color" attributes can be read from arrays 0 and 1.
	program.bindAttributeLocation("vertex", 0);
	program.bindAttributeLocation("color", 1);

	program.link();
	program.bind();

	// Get the memory position of the projection matrix in the shader program.
	// This will allow us to update the projection matrix faster (by index instead of by name).
	matrixLocationInShader = program.uniformLocation("projectionMatrix");

	// Prepare the vertex array object for first-time use
	visualizationData[Target_Grid] = new VisualizationData(f, GL_LINES);
	visualizationData[Target_Environment_Solid] = new VisualizationData(f, GL_TRIANGLES);
	visualizationData[Target_Environment_Contours] = new VisualizationData(f, GL_LINES);
	visualizationData[Target_Agents_Solid] = new VisualizationData(f, GL_TRIANGLES);
	visualizationData[Target_Agents_Contours] = new VisualizationData(f, GL_LINES);

	// initialize the simulation
	startNewSimulation(DefaultScenarioFile);

	program.release();
}

void UMANSOpenGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	program.bind();

	// Store the orthographic projection in our vertex shader
	QMatrix4x4 mat; mat.ortho(viewBounds);
	program.setUniformValue(matrixLocationInShader, mat);

	// draw everything
	for (auto vis : visualizationData)
		vis.second->Draw();

	program.release();
}

void UMANSOpenGLWidget::resizeGL(int w, int h)
{
	bool firstTime = !windowSize.isValid();

	if (!firstTime)
	{
		// calculate how much the window size has changed
		auto dx = (float)w - windowSize.width();
		auto dy = (float)h - windowSize.height();

		// how should the view bounds change then?
		auto scale = getScreenToWorldScale();
		dx *= scale.width();
		dy *= scale.height();

		viewBounds.adjust(-dx / 2.0, -dy / 2.0, dx / 2.0, dy / 2.0);
	}

	// store the new window size for later
	windowSize.setWidth(w);
	windowSize.setHeight(h);

	if (firstTime)
	{
		// set the view bounds for first time
		ZoomToFit();
	}
}

#pragma endregion

#pragma region [User interaction]

void UMANSOpenGLWidget::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::MouseButton::MiddleButton)
	{
		panning = true;
		// store the current mouse position, so that we can use it in a drag event
		m_lastPos = event->pos();

		updateCursor();
	}
}

void UMANSOpenGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	const auto& p = screenToWorld(event->pos());

	if (event->button() == Qt::MouseButton::MiddleButton)
	{
		panning = false;
		updateCursor();
	}

	else if (event->button() == Qt::MouseButton::LeftButton)
	{
		if (inAgentSelectMode)
		{
			// select the agent closest to the mouse position, if it is nearby enough
			const auto& nearest = simulator->GetWorld()->ComputeNeighbors(p, 50, nullptr);
			if (!nearest.first.empty())
			{
				// check the pixel distance between the mouse and the agent
				double worldDistance = distance(nearest.first[0].GetPosition(), p);
				double screenDistance = worldDistance / getScreenToWorldScale().width();

				// if the agent is close enough, select it
				if (screenDistance <= 50)
					setActiveAgent(simulator->GetWorld()->GetAgent(nearest.first[0].realAgent->getID()));

				// otherwise, don't select anything
				else
					setActiveAgent(nullptr);

			}

			// if there are no nearby agents at all, don't select anything
			else
				setActiveAgent(nullptr);
		}

		else
		{
			// add an agent at the mouse position
			Agent::Settings settings;
			settings.policy_ = simulator->GetPolicy(0);

			Agent* agent = simulator->GetWorld()->AddAgent(p, settings);
			agent->setGoal(p);
			activeAgent = agent;
		}

		drawSimulation();
		update();
	}

	else if (event->button() == Qt::MouseButton::RightButton)
	{
		// change the goal of the active agent
		if (activeAgent != nullptr)
		{
			activeAgent->setGoal(p);

			drawSimulation();
			update();
		}
	}
}

void UMANSOpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (panning)
	{
		// calculate how much the mouse has moved
		QPointF diffScreen(event->pos() - m_lastPos);

		// calculate how much the view bounds should be moved then
		auto scale = getScreenToWorldScale();
		QPointF diffWorld(diffScreen.x() * scale.width(), diffScreen.y() * scale.height());

		// move the view bounds by that amount
		viewBounds.adjust(-diffWorld.x(), -diffWorld.y(), -diffWorld.x(), -diffWorld.y());
	}

	// store the new mouse position for later
	m_lastPos = event->pos();

	// refresh the screen
	update();
}

void UMANSOpenGLWidget::wheelEvent(QWheelEvent* event)
{
	auto center = viewBounds.center();

	// zoom in or out
	float scale = (event->angleDelta().y() < 0 ? 1.1 : 0.9);
	viewBounds.setWidth(viewBounds.width() * scale);
	viewBounds.setHeight(viewBounds.height() * scale);

	// keep the center point the same
	viewBounds.moveCenter(center);

	// refresh the screen
	update();
}

void UMANSOpenGLWidget::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Key_SelectAgent)
	{
		inAgentSelectMode = true;
		updateCursor();
	}

	else if (event->key() == Key_PauseResume)
	{
		// pause or resume the simulation
		if (simulationRunning)
			PauseSimulation();
		else
			PlaySimulation();
	}

	else if (event->key() == Key_DeleteAgent)
	{
		// delete the selected agent
		if (activeAgent != nullptr)
		{
			simulator->GetWorld()->RemoveAgent(activeAgent->getID());
			setActiveAgent(nullptr);
			drawSimulation();
			update();
		}
	}

	else if (event->key() == Qt::Key::Key_S)
	{
		MakeScreenshot();
	}

	else if (event->key() == Qt::Key::Key_Escape)
	{
		// quit the application
		QCoreApplication::instance()->quit();
	}
}

void UMANSOpenGLWidget::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Key_SelectAgent)
	{
		inAgentSelectMode = false;
		updateCursor();
	}
}

void UMANSOpenGLWidget::setActiveAgent(Agent* agent)
{
	activeAgent = agent;
	activeAgentID = (agent == nullptr ? std::numeric_limits<size_t>::max() : activeAgent->getID());
}

void UMANSOpenGLWidget::checkActiveAgent()
{
	// if the agent with the desired ID no longer exists, reset the pointer to the selected agent
	if (activeAgentID != std::numeric_limits<size_t>::max() && simulator->GetWorld()->GetAgent(activeAgentID) == nullptr)
		setActiveAgent(nullptr);
}

#pragma endregion

#pragma region [Environment and simulation]

void UMANSOpenGLWidget::updateCursor()
{
	if (inAgentSelectMode)
		setCursor(Qt::PointingHandCursor);
	else if (panning)
		setCursor(Qt::OpenHandCursor);
	else
		setCursor(Qt::ArrowCursor);
}

void UMANSOpenGLWidget::updateSimulation()
{
	simulator->RunSimulationSteps(1);
	checkActiveAgent();
	drawSimulation();
	update();

	if (makeScreenshotsPerFrame)
		MakeScreenshot();

	// visualize the new simulation time
	updateSimulationTimerText();
}

void UMANSOpenGLWidget::updateSimulationTimerText()
{
	auto simulationTime = simulator->GetWorld()->GetCurrentTime();

	int totalMilliseconds = (int)round(1000 * simulationTime);

	int totalSeconds = totalMilliseconds / 1000;
	int totalMinutes = totalSeconds / 60;

	int millisecondsComponent = totalMilliseconds % 1000;
	int secondsComponent = totalSeconds % 60;
	int minutesComponent = totalMinutes % 60;

	std::string timeText =
		HelperFunctions::ToStringWithLeadingZeros(totalMinutes, 2) + ":"
		+ HelperFunctions::ToStringWithLeadingZeros(secondsComponent, 2) + "."
		+ HelperFunctions::ToStringWithLeadingZeros(millisecondsComponent, 3);

	mainApplication->GetUI().Label_SimulationTime->setText(QString(timeText.c_str()));
}

#pragma endregion