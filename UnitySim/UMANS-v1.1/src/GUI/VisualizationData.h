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

#ifndef VISUALIZATIONDATA_H
#define VISUALIZATIONDATA_H

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

/// <summary>A struct that can contain data to be shown on the screen. 
/// Used by the UMANSOpenGLWidget class in the UMANS GUI application.</summary>
struct VisualizationData
{
private:

	QOpenGLFunctions* openGLFunctions;
	QOpenGLVertexArrayObject vao;
	QOpenGLBuffer vbo_positions, vbo_colors;

	GLenum _drawMode;

	std::vector<QVector3D> vertices;
	std::vector<QColor> colors;

	bool hasChanged;
	bool isEnabled;

public:

	VisualizationData(QOpenGLFunctions* f, GLenum drawMode, bool enabled=true) 
		: openGLFunctions(f), _drawMode(drawMode), isEnabled(enabled), hasChanged(false)
	{
		vao.create();
		ClearData();
	}

	~VisualizationData()
	{
		vbo_positions.destroy();
		vbo_colors.destroy();
		vao.destroy();
	}

	void ToggleEnabled()
	{
		isEnabled = !isEnabled;
	}

	void AddData(const std::vector<QVector3D>& newPoints, const std::vector<QColor>& newColors)
	{
		vertices.insert(vertices.end(), newPoints.begin(), newPoints.end());
		colors.insert(colors.end(), newColors.begin(), newColors.end());
		hasChanged = true;
	}

	void AddData(const QVector3D& newPoint, const QColor& newColor)
	{
		vertices.push_back(newPoint);
		colors.push_back(newColor);
		hasChanged = true;
	}

	void ClearData()
	{
		vertices.clear();
		colors.clear();
	}

	void Draw()
	{
		if (!isEnabled)
			return;
		
		// if the data has changed since last time, refill the buffers
		if (hasChanged)
		{
			fillBuffers();
			hasChanged = false;
		}

		// draw the vertex/color data from the buffers
		QOpenGLVertexArrayObject::Binder vaoBinder(&vao);
		glDrawArrays(_drawMode, 0, (GLsizei)vertices.size());
	}

private:

	template<typename T> void fillSingleBuffer(QOpenGLBuffer& vbo,
		int arrayIndex, const std::vector<T>& list,
		const std::function<float(const T&)>& f1,
		const std::function<float(const T&)>& f2,
		const std::function<float(const T&)>& f3)
	{
		// create the vertex buffer object, if necessary
		if (!vbo.isCreated())
			vbo.create();
		vbo.bind();

		const size_t nrElements = list.size();
		const size_t elementSize = 3;
		GLfloat* data = new GLfloat[nrElements * 3];
		for (size_t i = 0; i < nrElements; ++i)
		{
			data[i * 3] = f1(list[i]);
			data[i * 3 + 1] = f2(list[i]);
			data[i * 3 + 2] = f3(list[i]);
		}
		vbo.allocate(data, (int)(nrElements * elementSize * sizeof(GLfloat)));

		delete[] data;

		// Store the vertex attribute bindings for the program.
		openGLFunctions->glEnableVertexAttribArray(arrayIndex);
		openGLFunctions->glVertexAttribPointer(arrayIndex, elementSize, GL_FLOAT, GL_FALSE, 0, 0);

		vbo.release();
	}

	void fillBuffers()
	{
		QOpenGLVertexArrayObject::Binder vaoBinder(&vao);

		// prepare the vertices
		fillSingleBuffer<QVector3D>(vbo_positions, 0, vertices, &QVector3D::x, &QVector3D::y, &QVector3D::z);

		// prepare the colors
		fillSingleBuffer<QColor>(vbo_colors, 1, colors, &QColor::redF, &QColor::greenF, &QColor::blueF);
	}

};

#endif //VISUALIZATIONDATA_H