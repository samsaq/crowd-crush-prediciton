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
** Authors: Julien Bruneau, Fabrice Atrévi
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;

/// <summary>
/// Main manager: Load the scene, create all the agents and manage users input
/// </summary>
public class LoadEnv : MonoBehaviour
{

#region attributes
    public List<GameObject> avatars;                        // List of agents
    public GameObject cam;                                  // Scene camera
    private CamRecorder cam_Rec;                            // Control scene recording
    CamMvt cam_Movement;                                    // Control camera movement behavior
    private FiltersControl cam_fc;                          // Control camera filters
    private MenuManager menuM;								// Control the starting menu
    private GameObject crowdParent;                         // Parent object for all agents
    //private float rotSpeed = 5;                             

    // --------------------------
    // CAMERA MOVEMENT PARAMETERS
    const float camTranslationSpeed = 25.0f;                // Camera translation speed
    const float camShiftPower = 250.0f;                     // Shift effect on camera speed
    const float camMaxShiftPower = 1000.0f;                 // Maximun shift effect on camera speed
    const float camRotationSpeed = 0.25f;                   // Rotation speed
    Vector3 lastMouse = new Vector3(255, 255, 255);         // Last mouse position to check its movement
    float camShiftHold = 1.0f;                              // Control the effect of shift with holding time  

    public static int cmptDone = 0;                         // Use to check if the last agent in the scene's animation is done
    public static int nbAgent = 0;                          // Store the number of agent in the scene

#endregion


    /// <summary>
    /// Scene and agents initialization
    /// </summary>
    void Start()
    {
        avatars = new List<GameObject>();
        cam_Rec = cam.GetComponent<CamRecorder>();
        cam_Movement = cam.GetComponent<CamMvt>();
        cam_fc = cam.GetComponent<FiltersControl>();

        cam_Rec.enabled = false;
        cam_Movement.enabled = false;

        menuM = gameObject.GetComponent<MenuManager>();
        menuM.CheckCommandLine();
    }

    /// <summary>
    /// Load a scenario: create stage and spawn agents
    /// </summary>
    /// <param name="trajDir">The path to the scenario file</param>
    public void loadScenario(string trajDir)
    {
        // --------------------------------------------------
        // SET RANDOM SEED TO HAVE SAME RESULTS AT EVERY RUNS
        Random.InitState(75482);

        // ---------------------------------
        // INITIALIZE SCENE FROM CONFIG FILE

        // -------------
        // CAMERA CONFIG
        cam_Rec.enabled                     = true;
        cam_Movement.enabled                = true;
        cam.transform.position              = ConfigReader.camPosition;
        cam.transform.rotation              = Quaternion.Euler(ConfigReader.camRotation);
        cam_Movement.lookAt_Id              = ConfigReader.camLookAtTarget;
        cam_Movement.follow_Id              = ConfigReader.camFollowTarget;
        cam_Movement.activateLookAt         = ConfigReader.lookAtAgent;
        cam_Movement.activateFollow         = ConfigReader.followAgent;
        cam_Movement.followOnX              = ConfigReader.camFollowOnX;
        cam_Movement.followOnY              = ConfigReader.camFollowOnY;
        cam_Movement.lockFirstPersonView    = ConfigReader.camLockFirstPerson;
        cam_Movement.smoothFirstPersonView  = ConfigReader.camSmoothFirstPerson;

        
        // -------------
        // RECORD CONFIG
        cam_Rec.record      = ConfigReader.recording;
        cam_Rec.timeToStart = ConfigReader.recordingStart;
        cam_Rec.timeToStop  = ConfigReader.recordingEnd;
        cam_Rec.framerate   = ConfigReader.recordingFramerate;
        cam_fc.init();

        //FB don't know why there this check
        //JB to see if it is a relative path or full one
        //FA don't work on my computer. Got trouble to save images. I change little bit to make it works. Have to check on windows
        if (ConfigReader.recordingSaveDir[0] == '.')
            cam_Rec.saveDir = Directory.GetCurrentDirectory().Replace("\\", "/") + "/" + ConfigReader.recordingSaveDir;
        else
            cam_Rec.saveDir = ConfigReader.recordingSaveDir;
        Camera.main.GetComponent<CamRecorder>().Init();
        InitDeepRecorders();

        // ---------------------
        // CLEAR PREVIOUS AGENTS
        foreach (GameObject a in avatars)
        {
            Destroy(a);
        }
        avatars.Clear();
        
        // -------------
        // CREATE AGENTS
        DirectoryInfo dir = new DirectoryInfo(trajDir);
        FileInfo[] info = dir.GetFiles("*.csv");
        if (info.Length == 0)
            info = dir.GetFiles("*.txt");

        if (crowdParent==null)
            crowdParent = new GameObject("Crowd");
        FollowTrajectory tmpFollower;
        
        int i = 0;
        object testRocketMan = Resources.Load("Prefabs/RocketBox/male/prefab_light/LS_m001_light");
        foreach (FileInfo f in info)
        {
            // ------------------------------------------------------------
            // IF NO ROCKETMAN (PROPRIETARY ASSETS), USE PLASTICMAN INSTEAD
            GameObject character = (testRocketMan!=null) ? CreateAgent_RocketBox(i) : CreateAgent_PlasticMan(i);
            character.transform.parent = crowdParent.transform;
            avatars.Add(character);
            // ---------------------
            // SET NAME, TAG, AND ID
            character.name = i.ToString("D4");
            character.tag = "Player";
            character.GetComponent<GetMemberPosition>().ID = i;

            // ----------------------------------
            // FOLLOWTRAJECTORY SCRIPT MANAGEMENT
            // Setup of the CSV filename and disable the start synchro
            tmpFollower = character.GetComponent<FollowTrajectory>();
            tmpFollower.SetCsvFilename(f.FullName);
            tmpFollower._SyncLaunchWithTrajectory = true; // start the character at the csv time


            // -------------------------------
            // Load animation controller from ressources
            //string PathAnimeController = "PlasticMan/Animation/LocomotionFinal";
            //Animator Anime = character.GetComponent<Animator>();
            //Anime.runtimeAnimatorController = Resources.Load(PathAnimeController) as RuntimeAnimatorController;
            //Anime.applyRootMotion = false;
            // make sure that object on the floor (height 0)
            character.transform.position = new Vector3(character.transform.position.x, 0.0f, character.transform.position.z);

            i++;
        }
        nbAgent = avatars.Count;
        cmptDone = 0;
        cam_Movement.initializeTargetObjects(avatars.ToArray());
        cam_fc.OnSceneChange();

    }

    /// <summary>
    /// Manage which deep recorder is used and initialize it
    /// </summary>
    private void InitDeepRecorders()
    {
        GameObject DeepRecorders = GameObject.FindGameObjectWithTag("DeepRecorders");
        if (ConfigReader.bboxeBody.record)
        {
            DeepRecorders.AddComponent<BodyBoundingBoxRecorder>();
            DeepRecorders.GetComponent<BodyBoundingBoxRecorder>().Init(ConfigReader.recordingSaveDir);
        }

        if (ConfigReader.bboxeHead.record)
        {
            DeepRecorders.AddComponent<HeadBoundingBoxRecorder>();
            DeepRecorders.GetComponent<HeadBoundingBoxRecorder>().Init(ConfigReader.recordingSaveDir);
        }
    }

    Color? GetColorForAgentId(int id)
    {
        if (ConfigReader.agentsColor != null && ConfigReader.agentsColor.Count > 0)
        {
            foreach (ConfigAgentColor c in ConfigReader.agentsColor)
            {
                if (id > c.firstAgent - 1 && id < c.lastAgent + 1 && (id - c.firstAgent) % c.step == 0)
                    return new Color(c.red, c.green, c.blue, 1);
            }
        }
        return null;
    }

    Material CreateMaterialWithColor(Color color)
    {
        var tmpMaterial = new Material(Shader.Find("Legacy Shaders/Diffuse"));
        tmpMaterial.color = color;
        return tmpMaterial;
    }

    string GetModelName_RocketBox(int id)
    {
        string path;
        int modelId;
        if (id % 2 == 0)
        {
            path = "male/prefab_light/LS_m";
            modelId = id / 2;
        }
        else
        {
            path = "female/prefab_light/LS_f";
            modelId = (id - 1) / 2;
        }
        modelId = (modelId % 20) + 1;
        return "Prefabs/RocketBox/" + path + modelId.ToString("D3") + "_light";
    }

    GameObject CreateAgent_RocketBox(int id)
    {
        // -------------------
        // LOAD AGENT IN UNITY
        GameObject rb = Instantiate(Resources.Load<GameObject>(GetModelName_RocketBox(id)));

        // -------------------
        // SET BOUNDING BOXES ITEM
        rb.AddComponent<GetMemberPositionRocketBox>();
        //gameObject.AddComponent<FollowTrajectory>();
        GameObject headBoundingBox = Instantiate(Resources.Load<GameObject>("HeadBoundingBoxes/Hbb_RocketBox"));
        Transform Head = rb.transform.Find("Bip01/Bip01 Pelvis/Bip01 Spine/Bip01 Spine1/Bip01 Spine2/Bip01 Neck/Bip01 Head");
        headBoundingBox.transform.parent = Head;

        // ---------------------------------------------
        // SET AGENT'S COLOR WHEN SPECIFY IN CONFIG FILE
        var tmpColor = GetColorForAgentId(id);
        if (tmpColor.HasValue)
        {
            var tmpMaterial = CreateMaterialWithColor(tmpColor.Value);
            Material[] mats = new Material[] { tmpMaterial, tmpMaterial, tmpMaterial };

            SkinnedMeshRenderer[] rendererList = rb.GetComponentsInChildren<SkinnedMeshRenderer>();
            foreach (SkinnedMeshRenderer renderer in rendererList)
                renderer.materials = mats;
        }

        return rb;
    }

    GameObject CreateAgent_PlasticMan(int id)
    {
        // -------------------
        // LOAD AGENT IN UNITY
        GameObject gameObject = Instantiate(Resources.Load<GameObject>("PlasticMan/plasticman"));

        // -------------------
        // SET BOUNDING BOXES ITEM
        gameObject.AddComponent<GetMemberPositionPlasticMan>();
        GameObject headBoundingBox = Instantiate(Resources.Load<GameObject>("HeadBoundingBoxes/Hbb_PlasticMan"));
        Transform Head = gameObject.transform.Find("Plasticman:Reference/Plasticman:Hips/Plasticman:Spine/Plasticman:Spine1/Plasticman:Spine2/Plasticman:Spine3/Plasticman:Neck/Plasticman:Head");
        headBoundingBox.transform.parent = Head;

        // -----------------
        // SET AGENT'S COLOR
        var tmpColor = GetColorForAgentId(id);
        if (!tmpColor.HasValue)
            tmpColor = new Color(Random.value, Random.value, Random.value, 1);

        var tmpRenderer = gameObject.GetComponentInChildren<SkinnedMeshRenderer>();
        tmpRenderer.material = CreateMaterialWithColor(tmpColor.Value);

        return gameObject;
    }

    /// <summary>
    /// Update camera state
    /// </summary>
    private void Update()
    {
        ConfigReader.camPosition = cam.transform.position;
        ConfigReader.camRotation = cam.transform.rotation.eulerAngles;
    }

    /// <summary>
    /// Manage users Input
    /// </summary>
    void LateUpdate()
    {
        // -------------------
        // ESCAPE => EXIT APPS
        if (Input.GetKeyDown(KeyCode.Escape) == true)
        {
            menuM.toogleMenu();
        }

        // -------------------------------
        // MOUSE + RCLICK => ROTATE CAMERA
        if (Input.GetMouseButton(1))
        {
            lastMouse = Input.mousePosition - lastMouse;
            lastMouse = new Vector3(-lastMouse.y * camRotationSpeed, lastMouse.x * camRotationSpeed, 0);
            lastMouse = new Vector3(cam.transform.eulerAngles.x + lastMouse.x, cam.transform.eulerAngles.y + lastMouse.y, 0);

            // Block all the way up or down to prevent glitches
            if (lastMouse.x > 180 && lastMouse.x < 270)
                lastMouse.x = 270;
            else if (lastMouse.x > 90 && lastMouse.x < 180)
                lastMouse.x = 90;

            cam.transform.eulerAngles = lastMouse;
        }
        lastMouse = Input.mousePosition;

        // -----------------------------------------
        // ARROWS, SHIFT, CTRL => CAMERA TRANSLATION
        Vector3 p = new Vector3();
        if (Input.GetKey(KeyCode.UpArrow))
        {
            p += new Vector3(0, 0, 1);
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            p += new Vector3(0, 0, -1);
        }
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            p += new Vector3(-1, 0, 0);
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            p += new Vector3(1, 0, 0);
        }
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            camShiftHold += Time.fixedUnscaledDeltaTime;
            p = p * Mathf.Max(camShiftHold * camShiftPower, camMaxShiftPower);
        }
        else
        {
            camShiftHold = Mathf.Clamp(camShiftHold * 0.5f, 1f, 1000f);
            p = p * camTranslationSpeed;
        }

        p = p * Time.fixedUnscaledDeltaTime;
        Vector3 newPosition = cam.transform.position;
        if (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl))
        { //If player wants to move on X and Z axis only
            cam.transform.Translate(p);
            newPosition.x = cam.transform.position.x;
            newPosition.z = cam.transform.position.z;
            cam.transform.position = newPosition;
        }
        else
        {
            cam.transform.Translate(p);
        }

        // --------------------------
        // F5 => SAVE CAMERA POSITION
        if (Input.GetKeyDown(KeyCode.F5))
        {
            menuM.saveConfig();
        }

        // ----------------------------
        // Cycle through filters F9 F10
        if (Input.GetKeyDown(KeyCode.F11))
        {
            cam_fc.cycleReset();
        }
        if (Input.GetKeyDown(KeyCode.F10))
        {
            cam_fc.cycleForward();
        }
        if (Input.GetKeyDown(KeyCode.F9))
        {
            cam_fc.cycleBackward();
        }

        // ----------------------
        // SPACE => PAUSE UNPAUSE
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (Time.timeScale == 0)
                menuM.Play();
            else
                menuM.Pause();
        }

        // --------------------------
        // p or P => Load the param setting panel
        if (Input.GetKeyDown("p"))
        {
            if((menuM.panel.activeSelf==false)&&(menuM.configPanel.activeSelf==false))
            {
                menuM.configParams();
            }
        }
    }

    /// <summary>
    /// Update the current scenario without destroy the previous one
    /// </summary>
    /// <param name="trajDir">The path to the scenario file</param>
    public void updateScenario(string trajDir)
    {
        // ---------------------------------
        // update SCENE FROM CONFIG FILE

        // -------------
        // CAMERA CONFIG
        cam_Rec.enabled                    = true;
        cam_Movement.enabled               = true;
        cam.transform.position             = ConfigReader.camPosition;
        cam.transform.rotation             = Quaternion.Euler(ConfigReader.camRotation);
        cam_Movement.lookAt_Id             = ConfigReader.camLookAtTarget;
        cam_Movement.follow_Id             = ConfigReader.camFollowTarget;
        cam_Movement.activateLookAt        = ConfigReader.lookAtAgent;
        cam_Movement.activateFollow        = ConfigReader.followAgent;
        cam_Movement.followOnX             = ConfigReader.camFollowOnX;
        cam_Movement.followOnY             = ConfigReader.camFollowOnY;
        cam_Movement.lockFirstPersonView   = ConfigReader.camLockFirstPerson;
        cam_Movement.smoothFirstPersonView = ConfigReader.camSmoothFirstPerson;


        // -------------
        // RECORD CONFIG
        cam_Rec.record          = ConfigReader.recording;
        cam_Rec.timeToStart     = ConfigReader.recordingStart;
        cam_Rec.timeToStop      = ConfigReader.recordingEnd;
        cam_Rec.framerate       = ConfigReader.recordingFramerate;
        cam_fc.init();

        if (ConfigReader.recordingSaveDir[0] == '.')
            cam_Rec.saveDir = Directory.GetCurrentDirectory().Replace("\\", "/") + "/" + ConfigReader.recordingSaveDir;
        else
            cam_Rec.saveDir = ConfigReader.recordingSaveDir;
        Camera.main.GetComponent<CamRecorder>().Init();
        InitDeepRecorders();

        cam_Movement.initializeTargetObjects(avatars.ToArray());
        cam_fc.OnSceneChange();
    }

}