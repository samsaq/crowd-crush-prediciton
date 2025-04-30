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

#ifndef LIB_COST_FUNCTION_PARAMETERS_H
#define LIB_COST_FUNCTION_PARAMETERS_H

#include <string>
#include <3rd-party/tinyxml/tinyxml2.h>

class Vector2D;

/// <summary>A helper class that represents an XML element with cost-function parameters. 
/// It has several functions for reading values of commonly-used types.</summary>
class CostFunctionParameters
{
private:
	/// <summary>A pointer to an XML element containing parameters for a CostFunction.</summary>
	const tinyxml2::XMLElement*  xml;
public:
	/// <summary>Creates a CostFunctionParameters wrapper for the given XML element.</summary>
	CostFunctionParameters(const tinyxml2::XMLElement* element) : xml(element) {}
	
	/// <summary>Tries to interpret an XML attribute as an integer.</summary>
	/// <param name="name">The name of the attribute to read from this XML element.</param>
	/// <param name="value">(out) Will store the integer value of the XML attribute with the given name.
	/// If the attribute does not exist, or if the value cannot be interpreted as an integer, then "value" remains unchanged.</param>
	/// <returns>true if the attribute was successfully interpreted and stored in "value"; 
	/// false otherwise, i.e. if the attribute does not exist, or if its value could not be interpreted as an integer.</returns>
	bool ReadInt(const std::string &name, int &value) const;

	/// <summary>Tries to interpret an XML attribute as a floating-point number.</summary>
	/// <param name="name">The name of the attribute to read from this XML element.</param>
	/// <param name="value">(out) Will store the floating-point value of the XML attribute with the given name.
	/// If the attribute does not exist, or if the value cannot be interpreted as a floating-point number, then "value" remains unchanged.</param>
	/// <returns>true if the attribute was successfully interpreted and stored in "value"; 
	/// false otherwise, i.e. if the attribute does not exist, or if its value could not be interpreted as a floating-point number.</returns>
	bool ReadFloat(const std::string &name, float &value) const;

	/// <summary>Tries to interpret an XML attribute as a Boolean value.</summary>
	/// <param name="name">The name of the attribute to read from this XML element.</param>
	/// <param name="value">(out) Will store the Boolean value of the XML attribute with the given name.
	/// If the attribute does not exist, or if the value cannot be interpreted as a Boolean value, then "value" remains unchanged.</param>
	/// <returns>true if the attribute was successfully interpreted and stored in "value"; 
	/// false otherwise, i.e. if the attribute does not exist, or if its value could not be interpreted as a Boolean value.</returns>
	bool ReadBool(const std::string &name, bool &value) const;

	/// <summary>Tries to obtain the string value of an XML attribute.</summary>
	/// <param name="name">The name of the attribute to read from this XML element.</param>
	/// <param name="value">(out) Will store the string value of the XML attribute with the given name.
	/// If the attribute does not exist, then "value" remains unchanged.</param>
	/// <returns>true if the attribute was successfully stored in "value"; 
	/// false otherwise, i.e. if the attribute does not exist.</returns>
	bool ReadString(const std::string &name, std::string &value) const;

	/// <summary>Tries to interpret two XML attributes together as a Vector2D.</summary>
	/// <param name="nameX">The name of attributes of this XML element that should store the x component of a vector.</param>
	/// <param name="nameY">The name of attributes of this XML element that should store the y component of a vector.</param>
	/// <param name="value">(out) Will store the floating-point values of the two attributes, wrapped in a Vector2D. 
	/// If either of these attributes does not exist or cannot be interpreted as a floating-point number, then "value" remains unchanged.</param>
	/// <returns>true if the attributes were successfully interpreted and stored in "value"; 
	/// false otherwise, i.e. if at least one attribute does not exist or could not be interpreted as a floating-point value.</returns>
	bool ReadVector2D(const std::string &nameX, const std::string &nameY, Vector2D &value) const;
};

#endif // LIB_COST_FUNCTION_PARAMETERS_H
