/* Crowd Simulator Engine
** Copyright (C) 2018 - Inria Rennes - Rainbow - Julien Pettre
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
**
** Authors: Julien Bruneau
**
** Contact: crowd_group@inria.fr
*/

#if (UNITY_EDITOR) 
using UnityEngine;
using System.Collections;
using UnityEditor;

/// <summary>
/// Add a button in the Editor to create a default config file with all configuration parameters
/// </summary>
[CustomEditor(typeof(LoadEnv))]
public class CustomEditor_LoadEnv : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if (GUILayout.Button("Create Template Config Files"))
        {
            ConfigReader.CreateTemplate();
            ObstaclesReader.CreateTemplate();
        }
    }
}
#endif
