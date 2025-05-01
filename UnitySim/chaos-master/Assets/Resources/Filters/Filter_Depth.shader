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

Shader "Filter/Filter_Depth"
{
	Properties
	{
	}

	SubShader{
		Tags { "RenderType" = "Opaque" }
		Pass {
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"

			struct v2f {
				float4 pos : SV_POSITION;
				float nz : TEXCOORD0;
				float4 worldPos : TEXCOORD2;
				UNITY_VERTEX_OUTPUT_STEREO
			};

			v2f vert(appdata_base v) {
				v2f o;
				UNITY_SETUP_INSTANCE_ID(v);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
				o.pos = UnityObjectToClipPos(v.vertex);
				o.nz = -(UnityObjectToViewPos(v.vertex).z);

				o.worldPos = mul(unity_ObjectToWorld, v.vertex);
				return o;
			}

			uniform float _maxDist;
			uniform float _minDist;
			uniform float _exponent;

			float4 frag(v2f i) : SV_Target {
				float depth = distance(i.worldPos, _WorldSpaceCameraPos) - _minDist;
				if (depth < 0)
					depth = 0;
				float ratio = _maxDist - _minDist;
				if (ratio <= 0)
					ratio = 1;
				depth = pow(depth / ratio, _exponent);

				return float4(depth, depth, depth, 1);
			}
			ENDCG
		}
	}
}
