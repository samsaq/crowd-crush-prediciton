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

using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;

public abstract class Filter
{
    // configuration
    public string name;
    public int quality;
    public bool supportsAntialiasing;
    public bool needsRescale;
    public int width=1920;
    public int height=1080;


    // impl
    public Shader filter;
    public Camera camera;

    public abstract void initParams();

    protected void createCam(Camera mainCam)
    {
        GameObject go = new GameObject(name, typeof(Camera));
        //go.hideFlags = HideFlags.HideAndDontSave;
        go.transform.parent = mainCam.transform;

        camera = go.GetComponent<Camera>();
    }

    public virtual void onCameraChange(Camera main, int display)
    {
        // cleanup capturing camera
        camera.RemoveAllCommandBuffers();

        // copy all "main" camera parameters into capturing camera
        camera.CopyFrom(main);

        camera.targetDisplay = display;

        camera.SetReplacementShader(filter, "");
        camera.backgroundColor = Color.black;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.allowHDR = false;
        camera.allowMSAA = false;

    }

    public virtual void OnSceneChange(Renderer r, ref MaterialPropertyBlock mpb)
    {
    }
}


public class FilterSegmentation : Filter
{
    public FilterSegmentation(Camera mainCam)
    {
        // configuration
        name = "Segmentation";
        supportsAntialiasing = false;
        needsRescale = false;

        // impl
        filter = Shader.Find("Filter/Filter_Segmentation");

        createCam(mainCam);
    }

    public override void initParams()
    {
        quality = ConfigReader.imgSegmentation.quality;
        width = ConfigReader.imgSegmentation.width;
        height = ConfigReader.imgSegmentation.height;
    }

    public override void OnSceneChange(Renderer r, ref MaterialPropertyBlock mpb)
    {
        var id = r.gameObject.GetInstanceID();
        mpb.SetColor("_ObjectColor", ColorEncoding.EncodeIDAsColor(id));
    }
}

public class FilterCategory : Filter
{
    public FilterCategory(Camera mainCam)
    {
        // configuration
        name = "Categories";
        supportsAntialiasing = false;
        needsRescale = false;

        // impl
        filter = Shader.Find("Filter/Filter_Category");

        createCam(mainCam);
    }

    public override void initParams()
    {
        quality = ConfigReader.imgCategories.quality;
        width = ConfigReader.imgCategories.width;
        height = ConfigReader.imgCategories.height;
    }

    public override void OnSceneChange(Renderer r, ref MaterialPropertyBlock mpb)
    {
        var layer = r.gameObject.layer;
        //var tag = r.gameObject.tag;
        mpb.SetColor("_CategoryColor", ColorEncoding.EncodeLayerAsColor(layer));
    }
}

public class FilterDepth : Filter
{
    public float maxDist = 50;
    public float minDist = 0;
    public float exponent = 1;

    public FilterDepth(Camera mainCam, float maxRepresentedDist=50f)
    {
        // configuration
        name = "Depth";
        supportsAntialiasing = true;
        needsRescale = false;
        maxDist = maxRepresentedDist;

        // impl
        filter = Shader.Find("Filter/Filter_Depth");

        createCam(mainCam);
    }

    public override void initParams()
    {
        quality = ConfigReader.imgDepth.quality;
        width = ConfigReader.imgDepth.width;
        height = ConfigReader.imgDepth.height;

        maxDist = ConfigReader.imgDepth.maxDepth;
        minDist = ConfigReader.imgDepth.minDepth;
        exponent = ConfigReader.imgDepth.exponent;
    }

    public override void OnSceneChange(Renderer r, ref MaterialPropertyBlock mpb)
    {
        var layer = r.gameObject.layer;
        //var tag = r.gameObject.tag;
        mpb.SetFloat("_maxDist", maxDist);
        mpb.SetFloat("_minDist", minDist);
        mpb.SetFloat("_exponent", exponent);
    }
}

public class FilterNormal : Filter
{
    public FilterNormal(Camera mainCam)
    {
        // configuration
        name = "Normals";
        supportsAntialiasing = true;
        needsRescale = false;

        // impl
        filter = Shader.Find("Filter/Filter_Normal");

        createCam(mainCam);
    }

    public override void initParams()
    {
        quality = ConfigReader.imgNormals.quality;
        width = ConfigReader.imgNormals.width;
        height = ConfigReader.imgNormals.height;
    }
}

public class FilterOpticalFlow : Filter
{
    public bool motionVector;


    int ncols = 0;
    Color[] colorwheel;


    private Material mat;
    float sensitivity = 20;

    public FilterOpticalFlow(Camera mainCam)
    {
        // configuration
        name = "OpticalFlow";
        supportsAntialiasing = false;
        needsRescale = true;

        // impl
        filter = Shader.Find("Filter/Filter_OpticalFlow");

        createCam(mainCam);
    }

    public override void initParams()
    {
        quality = ConfigReader.imgOpticalFlow.quality;
        width = ConfigReader.imgOpticalFlow.width;
        height = ConfigReader.imgOpticalFlow.height;

        motionVector = ConfigReader.imgOpticalFlow.motionVector;
        if (motionVector && !(quality == 16 || quality == 32))
            quality = 16;

    }

    public override void onCameraChange(Camera main, int display)
    {
        camera.RemoveAllCommandBuffers();

        camera.CopyFrom(main);

        camera.targetDisplay = display;

        camera.backgroundColor = Color.white;

        // cache materials and setup material properties
        if (!mat)
            mat = new Material(filter);
        if (ncols==0)
            makecolorwheel();


        mat.SetFloat("_Sensitivity", sensitivity);
        mat.SetColorArray("_ColorWheel", colorwheel);
        mat.SetInt("_ncols", ncols);
        mat.SetInt("_MotionVector", motionVector ? 1 : 0);

        var cb = new CommandBuffer();
        cb.Blit(null, BuiltinRenderTextureType.CurrentActive, mat);
        camera.AddCommandBuffer(CameraEvent.AfterEverything, cb);
        camera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;
    }

    // Color wheel
    void setcols(int r, int g, int b, int k)
    {
        colorwheel[k] = new Color(r / 255.0f, g / 255.0f, b / 255.0f);
    }

    void makecolorwheel()
    {
        // relative lengths of color transitions:
        // these are chosen based on perceptual similarity
        // (e.g. one can distinguish more shades between red and yellow 
        //  than between yellow and green)
        int RY = 15;
        int YG = 6;
        int GC = 4;
        int CB = 11;
        int BM = 13;
        int MR = 6;
        ncols = RY + YG + GC + CB + BM + MR;
        //printf("ncols = %d\n", ncols);

        colorwheel = new Color[ncols];

        int i;
        int k = 0;
        for (i = 0; i < RY; i++) setcols(255, 255 * i / RY, 0, k++);
        for (i = 0; i < YG; i++) setcols(255 - 255 * i / YG, 255, 0, k++);
        for (i = 0; i < GC; i++) setcols(0, 255, 255 * i / GC, k++);
        for (i = 0; i < CB; i++) setcols(0, 255 - 255 * i / CB, 255, k++);
        for (i = 0; i < BM; i++) setcols(255 * i / BM, 0, 255, k++);
        for (i = 0; i < MR; i++) setcols(255, 0, 255 - 255 * i / MR, k++);
    }
}
