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

using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

/// <summary>
/// Tools to serialize/deserialize data to/from XML files
/// </summary>
public static class XMLLoader {
    // -----------------------------------------
    // DATA SERIALIZATION FOR XML SAVING/LOADING
    #region dataSerialization
    public static void CreateXML<ObjType>(string fileName, object pObject)
    {
        StreamWriter writer;
        FileInfo t = new FileInfo(fileName);
        if (!t.Exists)
        {
            writer = t.CreateText();
        }
        else
        {
            t.Delete();
            writer = t.CreateText();
        }
        writer.Write(SerializeObject<ObjType>(pObject));
        writer.Close();
    }

    public static ObjType LoadXML<ObjType>(string fileName)
    {
        StreamReader r = File.OpenText(fileName);
        string _info = r.ReadToEnd();
        r.Close();

        if (_info.ToString() != "")
            return (ObjType)DeserializeObject<ObjType>(_info);
        else
            return default(ObjType);
    }

    static string SerializeObject<ObjType>(object pObject)
    {
        string XmlizedString = null;
        MemoryStream memoryStream = new MemoryStream();
        XmlSerializer xs = new XmlSerializer(typeof(ObjType));
        XmlTextWriter xmlTextWriter = new XmlTextWriter(memoryStream, Encoding.UTF8);

        xmlTextWriter.Formatting = Formatting.Indented;
        xs.Serialize(xmlTextWriter, pObject);
        memoryStream = (MemoryStream)xmlTextWriter.BaseStream;
        XmlizedString = UTF8ByteArrayToString(memoryStream.ToArray());
        return XmlizedString;
    }

    static object DeserializeObject<ObjType>(string pXmlizedString)
    {
        XmlSerializer xs = new XmlSerializer(typeof(ObjType));
        MemoryStream memoryStream = new MemoryStream(StringToUTF8ByteArray(pXmlizedString));
        return xs.Deserialize(memoryStream);
    }

    static string UTF8ByteArrayToString(byte[] characters)
    {
        UTF8Encoding encoding = new UTF8Encoding();
        string constructedString = encoding.GetString(characters);
        return (constructedString);
    }

    static byte[] StringToUTF8ByteArray(string pXmlString)
    {
        UTF8Encoding encoding = new UTF8Encoding();
        byte[] byteArray = encoding.GetBytes(pXmlString);
        return byteArray;
    }
    // DATA SERIALIZATION FOR XML SAVING/LOADING
    // -----------------------------------------
    #endregion
}
