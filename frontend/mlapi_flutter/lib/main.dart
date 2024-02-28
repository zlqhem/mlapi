import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  XFile? _image; //이미지를 담을 변수 선언
  String outputText = 'Ready';
  final ImagePicker picker = ImagePicker(); //ImagePicker 초기화

  Future getImage(ImageSource imageSource) async {
    print('getImage..');
    //pickedFile에 ImagePicker로 가져온 이미지가 담긴다.
    final XFile? pickedFile = await picker.pickImage(source: imageSource);

    if (pickedFile != null) {
      setState(() {
        _image = XFile(pickedFile.path);
      });
    }

    String msg = "";
    // Read bytes from the file object
    if (pickedFile != null) {
      try {
        Uint8List _bytes = await pickedFile.readAsBytes();

        // base64 encode the bytes
        String _base64String = base64.encode(_bytes);
        detect(_base64String);
        msg = "Done";
      } catch (e) {
        msg = "$e";
      }
    }

    setState(() {
      outputText = msg;
    });
  }

  void detect(String bytestring) async {
    setState(() {
      outputText = "Posting ...";
    });

    String endpoint =
        'https://qs9eqe954g.execute-api.us-east-1.amazonaws.com/detect';
    final detections = await http.post(
      Uri.parse(endpoint),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'image': bytestring,
      }),
    );

    setState(() {
      outputText = "Post processing ... $detections";
    });

    // TODO: post processing `detections` here
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text("Saltware: ML API Test v0.1")),
        body: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 30, width: double.infinity),
            _buildPhotoArea(),
            SizedBox(height: 20),
            _buildButton(),
            SizedBox(height: 20),
            _buildTextbox(),
            SizedBox(height: 10),
            _footer(),
          ],
        ),
      ),
    );
  }

  Widget _buildPhotoArea() {
    return _image != null
        ? Container(
            width: 300,
            height: 300,
            child: (kIsWeb)
                ? Image.network(_image!.path)
                : Image.file(File(_image!.path)), //가져온 이미지를 화면에 띄워주는 코드
          )
        : Container(
            width: 300,
            height: 300,
            child: Image.network(
                "https://images.unsplash.com/photo-1460088033389-a14158fa866d?q=80&w=640&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"),
            color: Colors.grey,
          );
  }

  Widget _buildButton() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(width: 30),
        ElevatedButton(
          onPressed: () {
            getImage(ImageSource.gallery); //getImage 함수를 호출해서 갤러리에서 사진 가져오기
          },
          child: Text("Gallery"),
        ),
      ],
    );
  }

  Widget _buildTextbox() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(width: 30),
        Text('$outputText'),
      ],
    );
  }

  Widget _footer() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(width: 30),
        SelectableText('http://github.com/zlqhem/mlapi'),
      ],
    );
  }
}
