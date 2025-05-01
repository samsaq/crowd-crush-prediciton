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
** Authors: Julien Bruneau, Fabrice Atrevi
**
** Contact: crowd_group@inria.fr
*/


using UnityEngine;
using System.IO;
using System.Xml;
using System.Xml.Serialization;
using System.Text;
using System.Collections.Generic;

/// <summary>
/// Static class used to load the simulation parameters from the Config.xml file.
/// </summary>
public static class ConfigReader
{
    #region privateStaticAttributs
    static string configFileName;
    static ConfigData data;         // Config data loaded from xml file
    #endregion

    /// <summary> 
    /// Manage configuration: Load XML file or create default one
    /// </summary>
    static ConfigReader()
    {
        string pathPlayer = Application.dataPath;
        int lastIndex = pathPlayer.LastIndexOf('/');
        string dataPath = pathPlayer.Remove(lastIndex, pathPlayer.Length - lastIndex);

        configFileName = dataPath + @" / Config.xml";
        data = new ConfigData();
    }

    /// <summary>
    /// Load a scenario file
    /// </summary>
    /// <param name="path">Path of the scenario file to load</param>
    static public void LoadConfig(string path)
    {
        configFileName = path;

        if (File.Exists(path))
        {
            data = XMLLoader.LoadXML<ConfigData>(path);
        }
    }

    /// <summary>
    /// Save current parameters in a scenario file
    /// </summary>
    /// <param name="path">Path of the scenario file to save</param>
    public static void SaveConfig(string path)
    {
        XMLLoader.CreateXML<ConfigData>(path, data);
    }

    /// <summary>
    /// Create a template config with all possible parameters
    /// </summary>
    public static void CreateTemplate()
    {
        string pathPlayer = Application.dataPath;
        int lastIndex = pathPlayer.LastIndexOf('/');
        string dataPath = pathPlayer.Remove(lastIndex, pathPlayer.Length - lastIndex);

        ConfigData template = new ConfigData();
        template.colorList = new List<ConfigAgentColor>();
        template.colorList.Add(new ConfigAgentColor(0, 1, 191, 1, 0, 0));
        template.colorList.Add(new ConfigAgentColor(192, 1, 383, 0, 1, 0));
        XMLLoader.CreateXML<ConfigData>(dataPath + @" / ConfigTemplate.xml", template);
    }

    #region get_set
    /// <summary>
    /// Folder of the trajectory files
    /// </summary>
    static public string trajectoriesDir
    {
        get { return data.env_filesPath; }
        set { data.env_filesPath = value; }
    }
    /// <summary>
    /// Data about an AssetBundle to load
    /// </summary>
    static public ConfigStage stageInfos
    {
        get { return data.env_stageInfos; }
        set { data.env_stageInfos = value; }
    }
    /// <summary>
    /// Path to the file containing the obstacles definition
    /// </summary>
    static public string obstaclesFile
    {
        get { return data.env_obstFile; }
        set { data.env_obstFile = value; }
    }
    /// <summary>
    /// camera Type
    /// </summary>
    static public int camType
    {
        get { return data.cam.cameraType.id; }
        set { data.cam.cameraType.id = (int)value; }
    }
    /// <summary>
    /// Starting camera position
    /// </summary>
    static public Vector3 camPosition
    {
        get { return data.cam.position.vect; }
        set { data.cam.position = new ConfigVect3(value); }
    }
    /// <summary>
    /// Starting camera rotation
    /// </summary>
    static public Vector3 camRotation
    {
        get { return data.cam.rotation.vect; }
        set { data.cam.rotation = new ConfigVect3(value); }
    }

    /// <summary>
    /// Boolean, true if the LookAtAgent id greater than -1
    /// </summary>
    static public bool lookAtAgent
    {
        get { return data.cam.lookAtAgent.id > -1; }
    }
    /// <summary>
    /// ID of the agent to look at
    /// </summary>
    static public int camLookAtTarget
    {
        get { return data.cam.lookAtAgent == null ? -1 : data.cam.lookAtAgent.id; }
        set { data.cam.lookAtAgent.id = value;}
    }
    /// <summary>
    /// Boolean, true if the FollowAgent id greater than -1
    /// </summary>
    static public bool followAgent
    {
        get { return data.cam.followAgent.id > -1; }
    }
    /// <summary>
    /// ID of the agent to follow with the camera
    /// </summary>
    static public int camFollowTarget
    {
        get { return data.cam.followAgent == null ? -1 : data.cam.followAgent.id; }
        set { data.cam.followAgent.id = value;}
    }
    /// <summary>
    /// Boolean, true if camera follow an agent's translation of the axe X
    /// </summary>
    static public bool camFollowOnX
    {
        get { return data.cam.followAgent == null ? false : data.cam.followAgent.followX; }
        set { data.cam.followAgent.followX = value;}
    }
    /// <summary>
    /// Boolean, true if camera follow an agent's translation of the axe Y
    /// </summary>
    static public bool camFollowOnY
    {
        get { return data.cam.followAgent == null ? false : data.cam.followAgent.followY; }
        set { data.cam.followAgent.followY = value;}
    }
    /// <summary>
    /// Boolean, true if camera first person or torsum is stuck with the agent direction or free to move
    /// </summary>
    static public bool camLockFirstPerson
    {
        get { return data.cam.followAgent == null ? false : data.cam.followAgent.lockFirstPerson; }
        set { data.cam.followAgent.lockFirstPerson = value; }
    }
    /// <summary>
    /// Boolean, if true the movment of camera first person or torsum is smooth, if false it oscillate with the head/torsum 
    /// </summary>
    static public bool camSmoothFirstPerson
    {
        get { return data.cam.followAgent == null ? false : data.cam.followAgent.smoothFirstPerson; }
        set { data.cam.followAgent.smoothFirstPerson = value; }
    }
    /// <summary>
    /// Boolean, true if the animation should be recorded
    /// </summary>
    static public bool recording
    {
        get { return data.recording.end > data.recording.start; }
    }
    /// <summary>
    /// time to start the recording of the animation
    /// </summary>
    static public float recordingStart
    {
        get { return data.recording.start; }
        set { data.recording.start = value;}
    }
    /// <summary>
    /// Time to stop the recording of the animation
    /// </summary>
    static public float recordingEnd
    {
        get { return data.recording.end; }
        set { data.recording.end = value;}
    }
    /// <summary>
    /// Framerate used for the recording of the animation
    /// </summary>
    static public int recordingFramerate
    {
        get { return data.recording.framerate; }
        set { data.recording.framerate = value;}
    }
    /// <summary>
    /// Folder where all the images from the animation are recorded
    /// </summary>
    static public string recordingSaveDir
    {
        get { return data.recording.saveDir; }
        set { data.recording.saveDir = value;}
    }
    /// <summary>
    /// Width of the image resolution
    /// </summary>
    static public int recordingWidth
    {
        get { return data.recording.width; }
        set { data.recording.width = value;}
    }
    /// <summary>
    /// Height of the image resolution
    /// </summary>
    static public int recordingHeight
    {
        get { return data.recording.height; }
        set { data.recording.height = value;}
    }

    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgOriginal imgOriginal
    {
        get { return data.recording.saveImgOriginal; }
        set { data.recording.saveImgOriginal= value;}
    }
    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgSegmentation imgSegmentation
    {
        get { return data.recording.saveImgSegmentation; }
        set { data.recording.saveImgSegmentation = value;}
    }
    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgCategories imgCategories
    {
        get { return data.recording.saveImgCategories; }
        set { data.recording.saveImgCategories = value;}
    }
    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgDepth imgDepth
    {
        get { return data.recording.saveImgDepth; }
        set { data.recording.saveImgDepth = value;}
    }
    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgNormals imgNormals
    {
        get { return data.recording.saveImgNormals; }
        set { data.recording.saveImgNormals = value;}
    }
    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataImgOpticalFlow imgOpticalFlow
    {
        get { return data.recording.saveImgOpticalFlow; }
        set { data.recording.saveImgOpticalFlow = value;}
    }

    /// <summary>
    /// Indicate if body bounding boxes should be recorded
    /// </summary>
    static public DataBodyBoundingBoxes bboxeBody
    {
        get { return data.recording.saveBodyBoundingBoxes; }
        set { data.recording.saveBodyBoundingBoxes = value;}
    }

    /// <summary>
    /// Indicate if head bounding boxes should be recorded
    /// </summary>
    static public DataHeadBoundingBoxes bboxeHead
    {
        get { return data.recording.saveHeadBoundingBoxes; }
        set { data.recording.saveHeadBoundingBoxes = value;}
    }
    
    /// <summary>
    /// List of colors for the agents
    /// </summary>
    static public List<ConfigAgentColor> agentsColor
    {
        get { return data.colorList; }
    }
    #endregion

}


#region XMLConfigClasses
/// <summary>
/// Main config class to be serialize in XML config
/// </summary>
public class ConfigData
{

    // Evironnement config
    public string env_filesPath;
    public string env_obstFile;
    public ConfigStage env_stageInfos;

    // Camera config
    public ConfigCam cam;


    // Record config
    public ConfigRecording recording;

    // Agent color setup
    [XmlArray("AgentColorList"), XmlArrayItem("color")]
    public List<ConfigAgentColor> colorList;

    public ConfigData()
    {
        // Evironnement config
        env_filesPath = ".\\TrajExample\\ExampleTwoColor\\";
        env_obstFile = "";
        env_stageInfos = new ConfigStage();

        // Camera config
        cam = new ConfigCam();

        // Record config
        recording = new ConfigRecording();

        // Agent color
        //colorList = new List<ConfigAgentColor>();
        //colorList.Add(new ConfigAgentColor(0, 1, 191, 1, 0, 0));
        //colorList.Add(new ConfigAgentColor(192, 1, 383, 0, 1, 0));
    }


}

/// <summary>
/// Parameters of a stage to load from an AssetBundle
/// </summary>
public class ConfigStage
{
    [XmlAttribute]
    public string stageName;
    public string file;
    public ConfigVect3 position;
    public ConfigVect3 rotation;

    public ConfigStage()
    {
        stageName = "";
        file = "";
        position = new ConfigVect3();
        rotation = new ConfigVect3();
    }
}

/// <summary>
/// Camera configuraton to be serialize in XML config
/// </summary>
public class ConfigCam
{
    public ConfigCamType cameraType;
    public ConfigVect3 position;
    public ConfigVect3 rotation;

    public ConfigCamBehaviour1 lookAtAgent;
    public ConfigCamBehaviour2 followAgent;

    public ConfigCam()
    {
        position = new ConfigVect3();
        rotation = new ConfigVect3();
        lookAtAgent = new ConfigCamBehaviour1();
        followAgent = new ConfigCamBehaviour2();
        cameraType = new ConfigCamType();
    }
}

/// <summary>
/// Vector 2D that can be serialize in XML config
/// </summary>
public class ConfigVect2
{
    [XmlAttribute]
    public float x;
    [XmlAttribute]
    public float y;

    public ConfigVect2()
    {
        x = 0;
        y = 0;
    }

    public ConfigVect2(Vector2 vect)
    {
        x = vect.x;
        y = vect.y;
    }

    public Vector2 vect
    {
        get { return new Vector2(x, y); }
    }
}

/// <summary>
/// Vector 3D that can be serialize in XML config
/// </summary>
public class ConfigVect3
{
    [XmlAttribute]
    public float x;
    [XmlAttribute]
    public float y;
    [XmlAttribute]
    public float z;

    public ConfigVect3()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    public ConfigVect3(Vector3 vect)
    {
        x = vect.x;
        y = vect.z;
        z = vect.y;
    }

    public Vector3 vect
    {
        get { return new Vector3(x, z, y); }
    }
}
public class ConfigCamType
{
    [XmlAttribute("typeID")]
    public int id;

    public ConfigCamType()
    {
        id = 0;
    }
}
/// <summary>
/// Camera behavior configuration to be serialize in XML config
/// </summary>
public class ConfigCamBehaviour1
{
    [XmlAttribute("agentID")]
    public int id;

    public ConfigCamBehaviour1()
    {
        id = -1;
    }
}

/// <summary>
/// Camera behavior configuration to be serialize in XML config
/// </summary>
public class ConfigCamBehaviour2
{
    [XmlAttribute("agentID")]
    public int id;
    [XmlAttribute("followOnX")]
    public bool followX;
    [XmlAttribute("followOnY")]
    public bool followY;
    [XmlAttribute("lockFirstPerson")]
    public bool lockFirstPerson;
    [XmlAttribute("smoothFirstPerson")]
    public bool smoothFirstPerson;

    public ConfigCamBehaviour2()
    {
        id = -1;
        followX = false;
        followY = false;
        lockFirstPerson = false;
        smoothFirstPerson = true;
    }
}

/// <summary>
/// Recording configuration to be serialize in XML config
/// </summary>
public class ConfigRecording
{
    [XmlAttribute]
    public float start;
    [XmlAttribute]
    public float end;
    [XmlAttribute]
    public int framerate;
    [XmlAttribute]
    public int width;
    [XmlAttribute]
    public int height;

    public DataImgOriginal saveImgOriginal;
    public DataImgSegmentation saveImgSegmentation;
    public DataImgCategories saveImgCategories;
    public DataImgDepth saveImgDepth;
    public DataImgNormals saveImgNormals;
    public DataImgOpticalFlow saveImgOpticalFlow;
    public DataBodyBoundingBoxes saveBodyBoundingBoxes;
    public DataHeadBoundingBoxes saveHeadBoundingBoxes;

    public string saveDir;



    public ConfigRecording()
    {
        start = 0;
        end = 0;
        framerate = 15;
        width = -1;
        height = -1;

        saveDir = ".\\Output\\";

        saveImgOriginal = new DataImgOriginal();
        saveImgSegmentation = new DataImgSegmentation();
        saveImgCategories = new DataImgCategories();
        saveImgDepth = new DataImgDepth();
        saveImgNormals = new DataImgNormals();
        saveImgOpticalFlow = new DataImgOpticalFlow();
        saveBodyBoundingBoxes = new DataBodyBoundingBoxes();
        saveHeadBoundingBoxes = new DataHeadBoundingBoxes();
    }
}

public class DataImgOriginal 
{
    [XmlAttribute]
    public bool record = true;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
}
public class DataImgSegmentation 
{
    [XmlAttribute]
    public bool record = false;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
}
public class DataImgCategories 
{
    [XmlAttribute]
    public bool record = false;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
}

public class DataImgDepth
{
    [XmlAttribute]
    public bool record = false;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
    [XmlAttribute]
    public float minDepth=0;
    [XmlAttribute]
    public float maxDepth=50;
    [XmlAttribute]
    public float exponent = 1;
}

public class DataImgNormals 
{
    [XmlAttribute]
    public bool record = false;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
}

public class DataImgOpticalFlow 
{
    [XmlAttribute]
    public bool record = false;
    [XmlAttribute]
    public int quality = 8;
    [XmlAttribute]
    public int width = -1;
    [XmlAttribute]
    public int height = -1;
    [XmlAttribute]
    public bool motionVector = false;

}

public class DataBodyBoundingBoxes 
{
    [XmlAttribute]
    public bool record = false;
}

public class DataHeadBoundingBoxes 
{
    [XmlAttribute]
    public bool record = false;
}

/// <summary>
/// Agents color configuration to be serialize in XML config
/// </summary>
public class ConfigAgentColor
{
    [XmlAttribute]
    public int firstAgent;
    [XmlAttribute]
    public int step;
    [XmlAttribute]
    public int lastAgent;

    [XmlAttribute]
    public float red;
    [XmlAttribute]
    public float green;
    [XmlAttribute]
    public float blue;

    public ConfigAgentColor()
    {
        firstAgent = 0;
        step = 1;
        lastAgent = -1;

        red = 1;
        green = 0;
        blue = 0;
    }

    public ConfigAgentColor(int f, int s, int l, float r, float g, float b)
    {
        firstAgent = f;
        step = s;
        lastAgent = l;

        red = r;
        green = g;
        blue = b;
    }
}
#endregion