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
** Authors: Florian Berton
**
** Contact: crowd_group@inria.fr
*/
using UnityEngine;
using System.IO;

public class CommandLineParser
{
    #region attribute
    protected string _ScenarioFile = "";
    protected Vector2 _CamResolution = new Vector2(-1,-1);
    protected int _Height = -1;
    #endregion

    #region getter
    /// <summary>
    /// Getter for the path to the scenario file
    /// </summary>
    public string GetScenarioFile()
    {
        return _ScenarioFile;
    }

    /// <summary>
    /// Getter for the resolution of the camera recorder
    /// </summary>
    public Vector2 GetCamResolution()
    {
        return _CamResolution;
    }
    #endregion

    #region function
    /// <summary>
    /// Basic Constructor
    /// </summary>
    public CommandLineParser()
    {

    }

    /// <summary>
    /// Constructor with an input command line
    /// </summary>
    public CommandLineParser(string[] Args)
    {
        ParseCommandLine(Args);
    }

    /// <summary>
    /// Parse the arguments given in the command line
    /// </summary>
    protected void ParseCommandLine(string[] Args)
    {
        int nb = 0;
        while (nb < Args.Length)
        {
            switch (Args[nb])
            {
                // check if it's a scenario file
                case "-s":
                    if (nb<Args.Length-1)
                    {
                        string ScenarioFolder = Application.dataPath;
                        int lastIndex = ScenarioFolder.LastIndexOf('/');
                        ScenarioFolder = ScenarioFolder.Remove(lastIndex, ScenarioFolder.Length - lastIndex);
                        string ScenarioPath = Args[++nb];
                        ScenarioPath = ScenarioPath.Replace("\\", "/");
                        _ScenarioFile = ScenarioFolder + '/' + ScenarioPath;
                        if (!System.IO.File.Exists(_ScenarioFile))
                            Application.Quit();
                    }
                    break;
                // check if it's the camera resolution
                case "-r":
                    if (nb < Args.Length - 2)
                    {
                        _CamResolution = new Vector2(int.Parse(Args[++nb]), int.Parse(Args[++nb]));
                    }
                    break;
            }
            nb++;
        }
    }
    #endregion
}