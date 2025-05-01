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
** Authors: Julien Bruneau, Florian Berton
**
** Contact: crowd_group@inria.fr
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Xml;
using System.Xml.Serialization;

/// <summary>
/// Load and create obstacles from an XML file and/or stage from an AssetBundle
/// </summary>
public class ObstaclesReader
{

    XMLObstacles obst;          // list of XML obstacle objects
    GameObject defaultWall;     // default wall used as template to create new one
    GameObject defaultPillar;   // default cylinder used as template to create new one
    GameObject ground;          // default ground used as template to create new one

    List<GameObject> obstList;  // list of created obstacles

    /// <summary>
    /// Find the tamplate obstacles and deactivate them
    /// </summary>
    public ObstaclesReader()
    {
        defaultPillar = null;
        defaultWall = null;
        obst = new XMLObstacles();
        obstList = new List<GameObject>();

        GameObject[] obstGO = GameObject.FindGameObjectsWithTag("Obst");
        foreach (GameObject g in obstGO)
        {
            if (g.name == "Wall")
                defaultWall = g;
            else if (g.name == "Pillar")
                defaultPillar = g;
            else if (g.name == "Ground")
                ground = g;

            g.SetActive(false);
        }
    }

    /// <summary>
    /// destroy all the created obstacles
    /// </summary>
    public void clear()
    {
        foreach (GameObject g in obstList)
            GameObject.Destroy(g);
        obstList.Clear();

    }

    /// <summary>
    /// Load data from an obstacle file and create them and load a stage from an AssetBundle
    /// </summary>
    /// <param name="file">path to the obstacle file</param>
    /// <param name="stageInfo">data about the AssetBundle</param>
    public void createObstacles(string file, ConfigStage stageInfo)
    {
        // --------------
        // LOAD OBSTACLES
        if (file != "")
        {
            obst = XMLLoader.LoadXML<XMLObstacles>(file);
            foreach (XMLRect g in obst.rectangles)
                createRectangle(g);
            foreach (XMLCylinder g in obst.cylinders)
                createCylinder(g);
        }
        // ----------
        // LOAD STAGE (and remove the ground if a stage is loaded)
        if (stageInfo.file != "" && stageInfo.stageName != "")
        {
            AssetBundle myLoadedAssetBundle = AssetBundle.LoadFromFile(stageInfo.file);
            if (myLoadedAssetBundle == null)
            {
                Debug.Log("Failed to load AssetBundle!");
                return;
            }
            GameObject prefab = myLoadedAssetBundle.LoadAsset<GameObject>(stageInfo.stageName);
            GameObject stage = GameObject.Instantiate(prefab);
            stage.transform.position += stageInfo.position.vect;
            stage.transform.Rotate(stageInfo.rotation.vect);
            if (stage.layer == 0) // if default layer
                SetLayerRecursively(stage, 10);
            //    stage.layer = 10; // obstacle
            //foreach (Transform child in stage.transform)
            //    if (child.gameObject.layer == 0) // if default layer
            //        child.gameObject.layer = 10; // obstacle
            

            obstList.Add(stage);

            ground.SetActive(false);
            myLoadedAssetBundle.Unload(false);
        }
        else
        {
            ground.SetActive(true);
        }
    }

    private void SetLayerRecursively(GameObject obj, int newLayer)
    {
        if (null == obj)
        {
            return;
        }

        obj.layer = newLayer;

        foreach (Transform child in obj.transform)
        {
            if (null == child)
            {
                continue;
            }
            SetLayerRecursively(child.gameObject, newLayer);
        }
    }


    /// <summary>
    /// Create a rectangular shaped obstacle
    /// </summary>
    /// <param name="infos">parameters of the obstacle to create</param>
    private void createRectangle(XMLRect infos)
    {
        float height = 1.8f;
        GameObject cube = GameObject.Instantiate(defaultWall);
        cube.transform.position = ((Vector3)infos.a + (Vector3)infos.b) / 2 + new Vector3(0, height / 2, 0);

        Vector3 dir = ((Vector3)infos.b - (Vector3)infos.a);
        //cylinder.transform.rotation.SetLookRotation(dir);
        cube.transform.LookAt((Vector3)infos.b + dir);
        cube.transform.rotation = Quaternion.Euler(0,cube.transform.rotation.eulerAngles.y,0);
        cube.transform.localScale = new Vector3(infos.w, height, dir.magnitude);

        cube.SetActive(true);
        obstList.Add(cube);
    }

    /// <summary>
    /// Create a cylinder shaped obstacle
    /// </summary>
    /// <param name="infos">parameters of the obstacle to create</param>
    private void createCylinder(XMLCylinder infos)
    {
        float height = 1.8f;
        GameObject cylinder = GameObject.Instantiate(defaultPillar);

        cylinder.transform.position = infos.c + new Vector3(0, height / 2, 0);
        cylinder.transform.localScale = new Vector3(infos.r, height / 2, infos.r);

        cylinder.SetActive(true);
        obstList.Add(cylinder);
    }

    /// <summary>
    /// Create an Example of obstacle file
    /// </summary>
    public static void CreateTemplate()
    {
        string pathPlayer = Application.dataPath;
        int lastIndex = pathPlayer.LastIndexOf('/');
        string dataPath = pathPlayer.Remove(lastIndex, pathPlayer.Length - lastIndex);

        XMLObstacles template = new XMLObstacles();
        template.rectangles.Add(new XMLRect(new XMLVect(0, 0), new XMLVect(5, 1), 1));
        template.rectangles.Add(new XMLRect(new XMLVect(0, -2.2f), new XMLVect(5, -2.2f), 1));
        template.cylinders.Add(new XMLCylinder(new XMLVect(-2, -1), 1.5f));
        template.cylinders.Add(new XMLCylinder(new XMLVect(6, 0), .5f));
        XMLLoader.CreateXML<XMLObstacles>(dataPath + @" / ObstaclesTemplate.xml", template);
    }

}

/// <summary>
/// Class containing the paramters of all obstacles, to be serialize in XML config
/// </summary>
public class XMLObstacles
{
    [XmlArray("Rectangles")]
    [XmlArrayItem("Rectangle")]
    public List<XMLRect> rectangles;
    [XmlArray("Cylinders")]
    [XmlArrayItem("Cylinder")]
    public List<XMLCylinder> cylinders;

    public XMLObstacles()
    {
        rectangles = new List<XMLRect>();
        cylinders = new List<XMLCylinder>();
    }
}
/// <summary>
/// Rectangular shaped obstacle, to be serialize in XML config
/// </summary>
public class XMLRect
{
    [XmlElement("A")]
    public XMLVect a;
    [XmlElement("B")]
    public XMLVect b;
    [XmlAttribute("Width")]
    public float w;

    public XMLRect()
    {
        a = null;
        b = null;
        w = 0;
    }

    public XMLRect(XMLVect side1, XMLVect side2, float width)
    {
        a = side1;
        b = side2;
        w = width;
    }
}

/// <summary>
/// Cylinder shaped obstacle, to be serialize in XML config
/// </summary>
public class XMLCylinder
{
    [XmlElement("Center")]
    public XMLVect c;
    [XmlAttribute("Radius")]
    public float r;

    public XMLCylinder()
    {
        c = null;
        r = 0;
    }

    public XMLCylinder(XMLVect center, float radius)
    {
        c = center;
        r = radius;
    }
}

/// <summary>
/// Vector to be serialize in XML config
/// </summary>
public class XMLVect
{
    [XmlAttribute("X")]
    public float x;
    [XmlAttribute("Y")]
    public float y;

    public XMLVect()
    {
        x = 0;
        y = 0;
    }
    public XMLVect(float X, float Y)
    {
        x = X;
        y = Y;
    }

    public static implicit operator Vector3(XMLVect o)
    {
        return o == null ? new Vector3(0, 0, 0) : new Vector3(-o.x, 0, o.y);
    }

    public static implicit operator XMLVect(Vector3 o)
    {
        return o == null ? null : new XMLVect(-o.x, o.z);
    }
}

