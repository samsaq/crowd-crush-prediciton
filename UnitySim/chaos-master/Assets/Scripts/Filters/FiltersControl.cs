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
** Based on code from Unity-Technologies bitbucket (Image Synthesis for Machine Learning)
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;

[RequireComponent(typeof(Camera))]
public class FiltersControl : MonoBehaviour
{

    [Header("Save Image Capture")]
    public bool saveImage = false;
    public bool saveIdSegmentation = false;
    public bool saveLayerSegmentation = false;
    public bool saveDepth = false;
    public bool saveNormals = false;
    public bool saveOpticalFlow = false;
    public string filepath = "..\\Captures";
    public string filename = "test.png";

    private int currentCam = 0;

    // pass configuration
    private Filter[] filters;
    private Camera mainCam;


    // cached materials
    private Material opticalFlowMaterial;

    void Start()
    {
        mainCam = GetComponent<Camera>();
        filters = new Filter[5];
        filters[0] = new FilterSegmentation(mainCam);
        filters[1] = new FilterCategory(mainCam);
        filters[2] = new FilterDepth(mainCam);
        filters[3] = new FilterNormal(mainCam);
        filters[4] = new FilterOpticalFlow(mainCam);

        OnCameraChange();
        OnSceneChange();
    }

    public void init()
    {
        // Image
        saveImage = ConfigReader.imgOriginal.record;

        // Segmentation
        saveIdSegmentation = ConfigReader.imgSegmentation.record;

        // Categories
        saveLayerSegmentation = ConfigReader.imgCategories.record;

        // Depth
        saveDepth = ConfigReader.imgDepth.record;

        // Normals
        saveNormals = ConfigReader.imgNormals.record;

        // Optical Flow
        saveOpticalFlow = ConfigReader.imgOpticalFlow.record;

        foreach (Filter f in filters)
        {
            f.initParams();
        }


        OnSceneChange();
        OnCameraChange();
    }


    void LateUpdate()
    {
//#if UNITY_EDITOR
//        if (DetectPotentialSceneChangeInEditor())
            OnSceneChange();
//#endif // UNITY_EDITOR

        // @TODO: detect if camera properties actually changed
        OnCameraChange();
    }

    private void setOn(int display)
    {
        if (display == 0)
            mainCam.targetDisplay = 0;
        else
            filters[display - 1].camera.targetDisplay = 0;
    }

    private void setOff(int display)
    {
        if (display == 0)
            mainCam.targetDisplay = filters.Length+1;
        else
            filters[display - 1].camera.targetDisplay = display;
    }

    public void cycleReset()
    {
        int newCam = 0;
        Debug.Log(currentCam + " => " + newCam);
        //capturePasses[currentCam].camera.enabled = false;
        //capturePasses[newCam].camera.enabled = true;

        setOff(currentCam);
        setOn(newCam);

        currentCam = newCam;

    }

    public void cycleForward()
    {
        int newCam = (currentCam + 1) % (filters.Length+1);
        Debug.Log(currentCam + " => " + newCam);
        //capturePasses[currentCam].camera.enabled = false;
        //capturePasses[newCam].camera.enabled = true;

        setOff(currentCam);
        setOn(newCam);

        currentCam = newCam;
    }


    public void cycleBackward()
    {
        int newCam = currentCam - 1;
        if (newCam < 0)
            newCam = filters.Length;
        Debug.Log(currentCam + " => " + newCam);
        //capturePasses[currentCam].camera.enabled = false;
        //capturePasses[newCam].camera.enabled = true;

        setOff(currentCam);
        setOn(newCam);

        currentCam = newCam;
    }

    public void OnCameraChange()
    {
        int targetDisplay = 1;
        var mainCamera = GetComponent<Camera>();

        foreach (Filter f in filters)
        {
            if (targetDisplay == currentCam)
                f.onCameraChange(mainCamera, 0);
            else
                f.onCameraChange(mainCamera, targetDisplay);
            targetDisplay++;
        }

    }


    public void OnSceneChange()
    {
        var renderers = Object.FindObjectsOfType<Renderer>();
        var mpb = new MaterialPropertyBlock();
        foreach (var r in renderers)
        {
            foreach (Filter f in filters)
            {
                f.OnSceneChange(r, ref mpb);
            }
            r.SetPropertyBlock(mpb);
        }
    }



    public void Save(string filename, int width = -1, int height = -1, string path = "")
    {

        var filenameExtension = System.IO.Path.GetExtension(filename);
        if (filenameExtension == "")
            filenameExtension = ".png";
        var filenameWithoutExtension = Path.GetFileNameWithoutExtension(filename);

        var pathWithoutExtension = Path.Combine(path, filenameWithoutExtension);

        // execute as coroutine to wait for the EndOfFrame before starting capture
        Save(filenameWithoutExtension, filenameExtension, width, height, path);
        //StartCoroutine(
        //    WaitForEndOfFrameAndSave(pathWithoutExtension, filenameExtension, width, height, path));
    }

    //private IEnumerator WaitForEndOfFrameAndSave(string filenameWithoutExtension, string filenameExtension, int width, int height, string path)
    //{
    //    yield return new WaitForEndOfFrame();
    //    Save(filenameWithoutExtension, filenameExtension, width, height, path);
    //}

    private void Save(string filenameWithoutExtension, string filenameExtension, int width, int height, string path)
    {
        if (saveImage)
        {
            string subPath = Path.Combine(path, "Images");
            if (!System.IO.Directory.Exists(subPath))
                System.IO.Directory.CreateDirectory(subPath);

            if (ConfigReader.imgOriginal.width==-1 || ConfigReader.imgOriginal.height==-1)
                Save(mainCam, Path.Combine(subPath, filenameWithoutExtension), width, height, true, false, ConfigReader.imgOriginal.quality);
            else
                Save(mainCam, Path.Combine(subPath, filenameWithoutExtension), ConfigReader.imgOriginal.width, ConfigReader.imgOriginal.height, true, false, ConfigReader.imgOriginal.quality);
        }

        foreach (var f in filters)
        {
            // Perform a check to make sure that the capture pass should be saved
            if (
                (f.name == "Segmentation" && saveIdSegmentation)
                || (f.name == "Categories" && saveLayerSegmentation)
                || (f.name == "Depth" && saveDepth)
                || (f.name == "Normals" && saveNormals)
                || (f.name == "OpticalFlow" && saveOpticalFlow)
                )
            {
                string subPath = Path.Combine(path, f.name);
                if (!System.IO.Directory.Exists(subPath))
                    System.IO.Directory.CreateDirectory(subPath);

                if (f.width==-1 || f.height==-1)
                    Save(f.camera, Path.Combine(subPath, filenameWithoutExtension), width, height, f.supportsAntialiasing, f.needsRescale, f.quality);
                else
                    Save(f.camera, Path.Combine(subPath, filenameWithoutExtension), f.width, f.height, f.supportsAntialiasing, f.needsRescale, f.quality);
            }
        }
    }

    private void Save(Camera cam, string filename, int width, int height, bool supportsAntialiasing, bool needsRescale, int quality=8)
    {
        if (width <= 0 || height <= 0)
        {
            width = Screen.width;
            height = Screen.height;
        }

        var mainCamera = GetComponent<Camera>();
        var depth = 24;
        var format = RenderTextureFormat.ARGBFloat;
        var readWrite = RenderTextureReadWrite.Linear;
        var antiAliasing = (supportsAntialiasing) ? Mathf.Max(1, QualitySettings.antiAliasing) : 1;

        var finalRT =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var renderRT = (!needsRescale) ? finalRT :
            RenderTexture.GetTemporary(mainCamera.pixelWidth, mainCamera.pixelHeight, depth, format, readWrite, antiAliasing);
        var tex = new Texture2D(width, height, TextureFormat.RGBAFloat, true);

        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = renderRT;
        cam.targetTexture = renderRT;

        cam.Render();

        if (needsRescale)
        {
            // blit to rescale (see issue with Motion Vectors in @KNOWN ISSUES)
            RenderTexture.active = finalRT;
            Graphics.Blit(renderRT, finalRT);
            RenderTexture.ReleaseTemporary(renderRT);
        }

        // read offsreen texture contents into the CPU readable texture
        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();

        // encode texture into PNG
        byte[] bytes;
        if (quality==32)
        {
            bytes = tex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
            filename = filename + ".exr";
        }
        else if (quality==16)
        {
            bytes = tex.EncodeToEXR(Texture2D.EXRFlags.None);
            filename = filename + ".exr";
        }
        else
        {
            bytes = tex.EncodeToPNG();
            filename = filename + ".png";
        }
        File.WriteAllBytes(filename, bytes);

        // restore state and cleanup
        cam.targetTexture = prevCameraRT;
        RenderTexture.active = prevActiveRT;

        Object.Destroy(tex);
        RenderTexture.ReleaseTemporary(finalRT);
    }

#if UNITY_EDITOR
    private GameObject lastSelectedGO;
    private int lastSelectedGOLayer = -1;
    private string lastSelectedGOTag = "unknown";
    private bool DetectPotentialSceneChangeInEditor()
    {
        bool change = false;
        // there is no callback in Unity Editor to automatically detect changes in scene objects
        // as a workaround lets track selected objects and check, if properties that are 
        // interesting for us (layer or tag) did not change since the last frame
        if (UnityEditor.Selection.transforms.Length > 1)
        {
            // multiple objects are selected, all bets are off!
            // we have to assume these objects are being edited
            change = true;
            lastSelectedGO = null;
        }
        else if (UnityEditor.Selection.activeGameObject)
        {
            var go = UnityEditor.Selection.activeGameObject;
            // check if layer or tag of a selected object have changed since the last frame
            var potentialChangeHappened = lastSelectedGOLayer != go.layer || lastSelectedGOTag != go.tag;
            if (go == lastSelectedGO && potentialChangeHappened)
                change = true;

            lastSelectedGO = go;
            lastSelectedGOLayer = go.layer;
            lastSelectedGOTag = go.tag;
        }

        return change;
    }
#endif // UNITY_EDITOR
}

