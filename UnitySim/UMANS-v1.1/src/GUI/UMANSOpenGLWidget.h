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

#ifndef UMANSOPENGLWIDGET_H
#define UMANSOPENGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>

#include "VisualizationData.h"
#include <core/crowdSimulator.h>

class UMANSQtGuiApplication;

/// <summary>Contains the visualization and behavior of the UMANS GUI application. 
/// This class includes functions for drawing simulation details on the screen, 
/// and functions that respond to interaction with UI elements.</summary>
class UMANSOpenGLWidget : public QOpenGLWidget
{
	Q_OBJECT

public:
	UMANSOpenGLWidget(QWidget *parent = nullptr);
	~UMANSOpenGLWidget();

public slots:
	void PlaySimulation();
	void PauseSimulation();
	void ResetSimulation();
	void SetPlaybackMultiplier(int value);
	void OpenScenarioFileDialog();
	void ToggleCSVOutput();
	void ZoomToFit();
	void ToggleGrid();
	void ToggleShowCostFunction();
	void ToggleScreenshots();
	void MakeScreenshot();

private slots:
	void updateSimulation();

protected:
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int width, int height) override;

	// events
	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void wheelEvent(QWheelEvent *event) override;
	void keyPressEvent(QKeyEvent *event) override;
	void keyReleaseEvent(QKeyEvent *event) override;

private:
	void showErrorMessage(const std::string& title, const std::string& details) const;
	void startNewSimulation(const std::string& scenarioFilename);
	void resetVisualization(bool deleteCurrentVisualization);

	Vector2D screenToWorld(const QPoint& screenPoint) const;
	QSizeF getScreenToWorldScale() const;

	void drawSimulation();

	void drawAgent(Agent& agent);
	void drawAgentCostCircle(Agent& agent);
	void drawAgentCostCircle_AddPoint(const Vector2D& basePos, const std::pair<Vector2D, float>& data, const float minCost, const float maxCost);

	void drawEnvironment(const bool refresh = false);
	void drawGrid(const bool refresh = false);
	void addPointsToBuffer(const std::vector<Vector2D>& points, const QColor& color, const std::string& target, const double depth);
	void addSegmentsToBuffer(const std::vector<LineSegment2D>& segments, const QColor& color, const std::string& target, const double depth);
	void addContourToBuffer(const std::vector<Vector2D>& points, const QColor& color, const std::string& target, const double depth);

	void updateCursor();
	void updateSimulationTimerText();
	void setActiveAgent(Agent* agent);
	void checkActiveAgent();


private:
	std::map<std::string, VisualizationData*> visualizationData;

	QOpenGLShaderProgram program;
	QOpenGLFunctions *f;
	QTimer *simulationTimer;

	UMANSQtGuiApplication* mainApplication;

	CrowdSimulator* simulator;
	Agent* activeAgent;
	size_t activeAgentID;

	std::vector<std::string> settingsFiles;

	bool inAgentSelectMode;
	bool panning;
	bool simulationRunning;
	int playbackMultiplier = 1;

	bool makeScreenshotsPerFrame;
	bool writeCSVOutput;
	bool showCostFunction;

	/// <summary>The part of the world that is currently being shown in the OpenGL widget.</summary>
	QRectF viewBounds;

	/// <summary>The width and height of theis OpenGL widget, in pixels.</summary>
	QSizeF windowSize;

	/// <summary>The last screen position of the mouse.</summary>
	QPoint m_lastPos;

	int matrixLocationInShader;
	
};

#endif //UMANSOPENGLWIDGET_H
