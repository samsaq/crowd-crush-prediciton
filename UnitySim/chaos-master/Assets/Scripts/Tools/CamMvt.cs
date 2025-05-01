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
** Authors: Julien Bruneau, Tristan Le Bouffant, Alberto Jovane, Fabrice ATREVI
**
** Contact: crowd_group@inria.fr
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class CamMvt : MonoBehaviour
{
    #region camera modes name
    public const string DEFAULT = "Default";
    public const string FREE_MOVMENT = "Free_Movement";
    public const string FOLLOW = "Follow";
    public const string LOOK_AT = "Look_At";
    public const string FIRST_PERSON = "First_Person";
    public const string TORSUM = "Torsum";
    #endregion


    #region attributes
    // Parameters interface
    public GameObject panelMovCamera;            // panel with the camera parameters on the interface
    public Dropdown camSelect;                   // the downdrop of the camera parameters

    // Camera mode
    public string current_camera_mode = "empty"; // the current Mode 

    public Vector3 _default_position;            // default position defined by the config file
    public Vector3 _default_rotation;            // default rotation defined by the config file

    // Rotation - Looking at an agent
    public bool activateLookAt = false;          // if true: lock the view to look at the selected agent -  else : free movment
    public bool activateFollow = false;          // if true: follow the selected agent

    private int _previous_lookAt_Id;             // Id of the previous agent to look at
    public int lookAt_Id;                        // Id of the agent to look at
    private GameObject lookAt_Agent = null;      // Agent to look at

    // Translation - Following an agent
    public int follow_Id;                        // Id of the agent to follow or to 
    private int _previous_follow_Id;             // Id of the previous agent to follow
    private GameObject follow_Agent;             // Agent to follow
    private Vector3 follow_LastPosition;         // Last position of the agent to follow
    public bool followOnX;                       // Lock axe X of the camera during camera's translation if false
    public bool followOnY;                       // Lock axe Z of the camera during camera's translation if false

    // Agent list
    GameObject[] agentList;

    // Head and Torsum references 
    private Transform agent_head_transform;      // transform reference of the head of the current followID agent
    private string[] head_names = new string[] { "HEAD", "Head", "head" };   //names of the head transform 
    private Transform agent_torsum_transform;    // transform reference of the head of the torsum followID agent 
    private string[] torsum_names = new string[] { "HIPS", "Hips", "hips", "Bip01 Spine1" }; //names of the torsum transform
    private Transform agent_reference_transform; // transform reference of the head of the torsum followID agent

    public bool lockFirstPersonView = false;     // if true: lock the view - else (default) : the user can move the view (mouse right click + movment)
    public bool smoothFirstPersonView = true;    // if true: the movment of the camera is smoothed on the y - else : the camera move with the head or torsum  
    private float smooth_y_head;                 // the y value to keep in first person view
    private float smooth_y_torsum;               // the y value to keep in torsum view


    // values holding the updates at run time
    private Vector3 _position_follow;             // the current follow position
    private Vector3 _position_first_person;       // the current first person position
    private Vector3 _position_torsum;             // the current torsum position
    string [] arguments;

    //reset
    public bool reset_active = false;
    #endregion


    #region initialization functions
    ///--------------------------------------------
    ///----- initialization functions -------------
    ///--------------------------------------------
    void Start()
    {
        panelMovCamera = GameObject.Find("/controlCanvas/cameraMovPanel");
        camSelect      = panelMovCamera.transform.Find("cameraType").GetComponent<Dropdown>();
        arguments      = System.Environment.GetCommandLineArgs();
    }

    /// <summary>
    /// Initialize the targets of the camera movement with parameters of config file
    /// </summary>
    public void initializeTargetObjects(GameObject[] list)
    {
        agentList = list;

        //initial positions
        _default_position = transform.position;
        _default_rotation = transform.eulerAngles;

        // initialize the look at and follow target
        update_look_at_target();
        update_follow_target();

        // initialize the follow position with the starting camera position
        _position_follow = transform.position;
    }
    #endregion

    #region run time functions
    ///--------------------------------------------
    ///----- run - time functions -----------------
    ///--------------------------------------------

    /// <summary>
    /// Update the target of look at camera
    /// </summary>
    public void update_look_at_target()
    {
        if (activateLookAt && lookAt_Id >= 0 && agentList.Length > 0)
        {
            if (lookAt_Id > agentList.Length - 1)
                lookAt_Id = agentList.Length - 1;
            _previous_lookAt_Id = lookAt_Id;
            lookAt_Agent = agentList[lookAt_Id];
        }
        else
        {
            lookAt_Id = -1;
            _previous_lookAt_Id = lookAt_Id;
        }
    }
    /// <summary>
    /// Update the target of the follow camera and of the first_person/torsum camera
    /// </summary>
    public void update_follow_target()
    {
        if (activateFollow && follow_Id >= 0 && agentList.Length > 0)
        {
            if (follow_Id > agentList.Length - 1)
                follow_Id = agentList.Length - 1;
            _previous_follow_Id = follow_Id;
            follow_Agent = agentList[follow_Id];
            
            //initialize the positions
            follow_LastPosition = follow_Agent.transform.position;
            agent_head_transform = get_transform_reference_of_head(follow_Agent);
            agent_torsum_transform = get_transform_reference_of_hips(follow_Agent);
            agent_reference_transform = follow_Agent.transform;
        }
        else
        {
            follow_Id = -1;
            _previous_follow_Id = follow_Id;
        }
    }

    /// <summary>
    /// Reads the parameter of the Usear Interface at every frame
    /// </summary>
    public void readCameraMovParamOnScene()
    {          
        lookAt_Id             = int.Parse(panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text);
        follow_Id             = int.Parse(panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text);                 
        followOnX             = panelMovCamera.transform.Find("FollowX").GetComponent<Toggle>().isOn; 
        followOnY             = panelMovCamera.transform.Find("FollowY").GetComponent<Toggle>().isOn;
        lockFirstPersonView   = panelMovCamera.transform.Find("LockPerson").GetComponent<Toggle>().isOn;
        smoothFirstPersonView = panelMovCamera.transform.Find("SmoothView").GetComponent<Toggle>().isOn;
    }

    /// <summary>
    /// It detects changes in the Current Camera Mode and intialize the parameters accordingly at every frame
    /// </summary>
    private void check_changes_and_initialize()
    {
        //1) change of state
        if(current_camera_mode != camSelect.options[camSelect.value].text)
        {
            //PREVIOUS CAMERA deinitialization
            if (current_camera_mode == FREE_MOVMENT)
            {
                //
            }
            else if (current_camera_mode == FOLLOW)
            {
                activateFollow = false;
            }
            else if (current_camera_mode == LOOK_AT)
            {
                activateLookAt = false;
            }
            else if (current_camera_mode == FIRST_PERSON)
            {
                activateFollow = false;
            }
            else if (current_camera_mode == TORSUM)
            {
                activateFollow = false;
            }

            //update
            current_camera_mode = camSelect.options[camSelect.value].text;

            //NEW CAMERA initialization
            if(current_camera_mode == FREE_MOVMENT)
            {
                //
            }
            else if (current_camera_mode == FOLLOW)
            {
                //activate and set 0 id in case or not predefined id
                if (follow_Id < 0 && agentList.Length > 0)
                {
                    follow_Id = 0;
                    panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text = follow_Id.ToString();
                }
                activateFollow = true;

                //set default rotation
                //transform.eulerAngles = _default_rotation;
                _position_follow = Camera.main.transform.position;
            }
            else if (current_camera_mode == LOOK_AT)
            {
                //activate and set 0 id in case or not predefined id
                if (lookAt_Id < 0 && agentList.Length > 0)
                {
                    lookAt_Id = 0;
                    panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text = lookAt_Id.ToString();
                }
                activateLookAt = true;

                //set default position and rotation
                //transform.eulerAngles = _default_rotation;
                //transform.position = _default_position;

            }
            else if (current_camera_mode == FIRST_PERSON)
            {
                //activate and set 0 id in case or not predefined id
                if (follow_Id < 0 && agentList.Length > 0)
                {
                    follow_Id = 0;
                    panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text = follow_Id.ToString();
                }
                activateFollow = true;

                //set default rotation
                update_follow_target();
                transform.forward = agent_reference_transform.forward;

                //initialize smooth value
                smooth_y_head = agent_head_transform.position.y;
            }
            else if (current_camera_mode == TORSUM)
            {
                //activate and set 0 id in case or not predefined id
                if (follow_Id < 0 && agentList.Length > 0)
                {
                    follow_Id = 0;
                    panelMovCamera.transform.Find("AgentID").GetComponent<InputField>().text = follow_Id.ToString();
                }
                activateFollow = true;
                
                //set default rotation
                update_follow_target();
                transform.forward = agent_reference_transform.forward;

                //initalize smooth value
                smooth_y_torsum = agent_torsum_transform.position.y;
            }
        }

        //2) change of agent id
        if (follow_Id != _previous_follow_Id && activateFollow)
        {
            update_follow_target();
        }
        if (lookAt_Id != _previous_lookAt_Id && activateLookAt)
        {
            update_look_at_target();
        }
    }
    
    private void initialize_for_command_line()
    {
        //PREVIOUS CAMERA deinitialization
        
        if (current_camera_mode == FOLLOW)
        {
            activateFollow = false;
        }
        else if (current_camera_mode == LOOK_AT)
        {
            activateLookAt = false;
        }
        else if (current_camera_mode == FIRST_PERSON)
        {
            activateFollow = false;
        }
        else if (current_camera_mode == TORSUM)
        {
            activateFollow = false;
        }

        //NEW CAMERA initialization
        
        if (current_camera_mode == FOLLOW)
        {
            //activate and set 0 id in case or not predefined id
            if (follow_Id < 0 && agentList.Length > 0)
            {
                follow_Id = 0;
            }
            activateFollow = true;

            //set default rotation
            //transform.eulerAngles = _default_rotation;
            _position_follow = Camera.main.transform.position;
        }
        else if (current_camera_mode == LOOK_AT)
        {
            //activate and set 0 id in case or not predefined id
            if (lookAt_Id < 0 && agentList.Length > 0)
            {
                lookAt_Id = 0;
            }
            activateLookAt = true;

            //set default position and rotation
            //transform.eulerAngles = _default_rotation;
            //transform.position = _default_position;

        }
        else if (current_camera_mode == FIRST_PERSON)
        {
            //activate and set 0 id in case or not predefined id
            if (follow_Id < 0 && agentList.Length > 0)
            {
                follow_Id = 0;
            }
            activateFollow = true;

            //set default rotation
            update_follow_target();
            transform.forward = agent_reference_transform.forward;

            //initialize smooth value
            smooth_y_head = agent_head_transform.position.y;
        }
        else if (current_camera_mode == TORSUM)
        {
            //activate and set 0 id in case or not predefined id
            if (follow_Id < 0 && agentList.Length > 0)
            {
                follow_Id = 0;
            }
            activateFollow = true;
            
            //set default rotation
            update_follow_target();
            transform.forward = agent_reference_transform.forward;

            //initalize smooth value
            smooth_y_torsum = agent_torsum_transform.position.y;
        }
    }
    #endregion

    #region events functions
    ///--------------------------------------------
    ///-------------- event functions -------------
    ///--------------------------------------------
    public void reset_position()
    {
        reset_active = true;
    }

    /// <summary>
    /// chek for events every frame
    /// </summary>
    private void check_for_events()
    {
        if(reset_active)
        {
            if (current_camera_mode == FREE_MOVMENT)
            {
                transform.eulerAngles = _default_rotation;
                transform.position = _default_position;
            }
            if (current_camera_mode == FOLLOW)
            {
                transform.eulerAngles = _default_rotation;
            }
            if (current_camera_mode == LOOK_AT)
            {
                transform.eulerAngles = _default_rotation;
                transform.position = _default_position;
            }
            if(current_camera_mode == FIRST_PERSON)
            {
                transform.forward = agent_reference_transform.forward;
            }
            if (current_camera_mode == TORSUM)
            {
                transform.forward = agent_reference_transform.forward;
            }
            reset_active = false;
        }
    }
    #endregion

    public void camMovParamsCommandLine()
    {
        if(ConfigReader.camType == 0)
            current_camera_mode   = "Free_Movement";
        else if(ConfigReader.camType == 1)
            current_camera_mode   = "Follow";
        else if(ConfigReader.camType == 2)
            current_camera_mode   = "Look_At";
        else if(ConfigReader.camType == 3)
            current_camera_mode   = "First_Person";
        else if(ConfigReader.camType == 4)
            current_camera_mode   = "Torsum";

        lookAt_Id             = ConfigReader.camLookAtTarget;
        follow_Id             = ConfigReader.camFollowTarget;                 
        followOnX             = ConfigReader.camFollowOnX; 
        followOnY             = ConfigReader.camFollowOnY;
        lockFirstPersonView   = ConfigReader.camLockFirstPerson;
        smoothFirstPersonView = ConfigReader.camSmoothFirstPerson;
    }

    ///--------------------------------------------
    ///---------------- main loop -----------------
    ///--------------------------------------------
    void Update()
    {
        #region UPDATE VALUES
        /// ----------------------------
        /// UPDATE VALUES --------------
        /// ----------------------------
        if (arguments.Length >1 && !Application.isEditor)
        {
            camMovParamsCommandLine();
            initialize_for_command_line();
        }
        else
        {
            readCameraMovParamOnScene();
            check_changes_and_initialize();
        }
        
        check_for_events();
        #endregion

        #region ROTATION
        // ------------------------------
        // ROTATION - LOOKING AT AN AGENT
        if(current_camera_mode == DEFAULT)
        {
            transform.eulerAngles = _default_rotation;
        }
        if(current_camera_mode == LOOK_AT)
        {
            if (lookAt_Agent != null && activateLookAt)
                transform.LookAt(lookAt_Agent.transform);
        }
        if (current_camera_mode == FIRST_PERSON)
        {
            if (lockFirstPersonView)
                transform.forward = agent_reference_transform.forward;
        }
        else if (current_camera_mode == TORSUM)
        {
            if (lockFirstPersonView)
                transform.forward = agent_reference_transform.forward;
        }

        #endregion

        #region TRANSLATION
        // --------------------------------
        // TRANSLATION --------------------
        // --------------------------------
        if(current_camera_mode == LOOK_AT)
        {
            transform.position = new Vector3(transform.position.x, _default_position.y, transform.position.z);
        }
        if (follow_Agent != null && activateFollow)
        {
            Vector3 delta = follow_Agent.transform.position - follow_LastPosition;
            follow_LastPosition = follow_Agent.transform.position;
            if (!followOnX){
                delta.x = 0;
            }
            if (!followOnY){
                delta.z = 0;
            }
            _position_follow = _position_follow + delta;

            _position_first_person = agent_head_transform.position;

            _position_torsum = agent_torsum_transform.position;


            if (current_camera_mode == FOLLOW)
            {
                transform.position = _position_follow;
            }
            else if (current_camera_mode == FIRST_PERSON)
            {
                if (smoothFirstPersonView)
                    transform.position = new Vector3(_position_first_person.x, smooth_y_head, _position_first_person.z);
                else
                    transform.position = _position_first_person;
            }
            else if (current_camera_mode == TORSUM)
            {
                if (smoothFirstPersonView)
                    transform.position = new Vector3(_position_torsum.x, smooth_y_torsum, _position_torsum.z);
                else
                    transform.position = _position_torsum;
            }
        }
        #endregion
    }


    #region Additional Functions
    ///--------------------------------------------
    ///----- additional functions -----------------
    ///--------------------------------------------

    /// <summary>
    /// Find the head transform and return it
    /// </summary>
    /// <param name="agent">the agent gameObject</param>
    /// <returns> The reference of the trasform of the head if found, otherwise null </returns>
    private Transform get_transform_reference_of_head(GameObject agent, bool general_approach = true)
    {
        Transform result;

        if (general_approach)
        {
            //general approach work with all the 
            result = recursive_find_names(agent.transform, head_names);
        }
        else
        {
            //specific approach if the structure is different crash
            result = agent.transform.GetChild(0).GetChild(0).GetChild(2).GetChild(0).GetChild(0).GetChild(0).GetChild(1).GetChild(0);
        }
        return result;
    }


    /// <summary>
    /// Find the head transform and return it
    /// </summary>
    /// <param name="agent">the agent gameObject</param>
    /// <returns> The reference of the trasform of the head if found, otherwise null </returns>
    private Transform get_transform_reference_of_hips(GameObject agent, bool general_approach = true)
    {
        Transform result;
        if (general_approach)
        {
            //general approach work with all the 
            result = recursive_find_names(agent.transform, torsum_names);
        }
        else
        {
            //specific approach if the structure is different crash
            result = agent.transform.GetChild(0).GetChild(0);
        }
        return result;
    }

    /// <summary>
    /// recursive look for "names" in child hierrarchy 
    /// </summary>
    /// <param name="parent"> the transform, starting point of the look for head child </param>
    /// <param name="names"> the list of names to look for</param>
    /// <returns>The reference of the trasform of the head if found, otherwise null</returns>
    private Transform recursive_find_names(Transform parent, string[] names)
    {
        if (parent.childCount == 0)
            return null;
        else
        {
            foreach (Transform child in parent.transform)
            {
                bool contains = false;
                foreach (string name in names)
                {
                    contains = contains || child.name.Contains(name);
                }
                if (contains)
                    return child;
            }
            foreach (Transform child in parent.transform)
            {
                Transform result = recursive_find_names(child, names);
                if (result != null)
                    return result;
            }
            return null;
        }
    }
    #endregion
}