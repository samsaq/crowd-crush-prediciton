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

#ifndef COLOR_H
#define COLOR_H

/// A simple wrapper for a color with R, G, B, and A components between 0 and 255.
struct Color
{
	/// The red component of this Color (range 0-255).
	unsigned short r;
	/// The green component of this Color (range 0-255).
	unsigned short g;
	/// The blue component of this Color (range 0-255).
	unsigned short b;
	/// The alpha component of this Color (range 0-255).
	unsigned short a;

	/// Default constructor for Color; creates the color black.
	Color() : r(0), g(0), b(0), a(255) {}

	/// Creates a Color with the given R, G, B, and A components.
	Color(unsigned short r, unsigned short g, unsigned short b, unsigned short a=255) : r(r), g(g), b(b), a(a) {}

	/// <summary>Creates a Color based on HSV (Hue, Saturation, Value) values.</summary>
	/// <remarks>Source: https://gist.github.com/kuathadianto/.</remarks>
	/// <param name="H">The H (Hue) component of the color, between 0 and 360.</param>
	/// <param name="S">The S (Saturation) component of the color, between 0 and 1.</param>
	/// <param name="V">The V (Vaue) component of the color, between 0 and 1.</param>
	/// <returns>A Color with RGB values that constitute the same color as the given HSV values.</returns>
	static Color FromHSV(int H, double S, double V)
	{
		H = H % 360;

		double C = S * V;
		double X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
		double m = V - C;
		double Rs, Gs, Bs;

		if (H < 60) // red --> yellow
		{ 
			Rs = C;
			Gs = X;
			Bs = 0;
		}
		else if (H < 120) // yellow --> green
		{ 
			Rs = X;
			Gs = C;
			Bs = 0;
		}
		else if (H < 180) // green --> cyan
		{ 
			Rs = 0;
			Gs = C;
			Bs = X;
		}
		else if (H < 240) // cyan --> blue
		{ 
			Rs = 0;
			Gs = X;
			Bs = C;
		}
		else if (H < 300) // blue --> magenta
		{ 
			Rs = X;
			Gs = 0;
			Bs = C;
		}
		else // magenta --> red
		{ 
			Rs = C;
			Gs = 0;
			Bs = X;
		}

		return Color(
			(int)((Rs + m) * 255),
			(int)((Gs + m) * 255),
			(int)((Bs + m) * 255));
	}
};

#endif //COLOR_H