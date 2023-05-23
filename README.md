# 프로젝트 소개


<div style="text-align: center;">
<img style="max-width: 70%;"
    src="https://leeyeonjun85.github.io/assets/images/etc/project-2822430_1920.jpg"
    alt="image">
</div>


## 목차
- [프로젝트 소개](#프로젝트-소개)
  - [목차](#목차)
  - [📜 스토리](#-스토리)
  - [AI 모델 1 : 전복 연령 예측](#ai-모델-1--전복-연령-예측)
  - [AI 모델 2 : 펄사 별(Pulsar Star) 관측](#ai-모델-2--펄사-별pulsar-star-관측)
  - [AI 모델 3 : 철판 불량 검출](#ai-모델-3--철판-불량-검출)
  - [멤버 소개](#멤버-소개)
  - [git 관련 bash 명령어](#git-관련-bash-명령어)
  - [링크](#링크)


<br><br><hr>


## 📜 스토리
<div>
    <p>우리 회사 <code>AI Code</code>는 다양한 인공지능 모델을 개발하여 필요한 회사에 제공합니다.</p>
    <p>최근 AI Code는 3개 회사에서 <b>인공지능 모델 업그레이드</b> 업무를 수주하였습니다.</p>
    <p>모델업그레이드 뿐만 아니라 사용자들이 업그레이드 된 모델을 직접 경험할 수 있도록 <b>웹서비스를 제공</b>해야 합니다.</p>
</div>


<br><br><hr>


## AI 모델 1 : 전복 연령 예측
<div>
    <h5>1. 데이터</h5>
    <p>• 독립변수 : 성별, 길이, 둘레 등 8개 특성</p>
    <p>• 종속변수 : 전복 연령</p>
    <br>
    <h5>2. 기본 모델 성능 : 회귀</h5>
    <p>• Epoch : 10 , lr : 0.001</p>
    <p>• Train Loss / Accuracy : 5.863 / 0.824</p>
    <p>• Test Accuracy : 0.827</p>
    <br>
    <h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
    <div class="text-center indent_0"><a class="btn btn-outline-primary"
        href="{% url 'GPTeachersDay:eda_abalone' %}">
        EDA 전복 : 바로가기
    </a></div>
</div>


<br><br><hr>


## AI 모델 2 : 펄사 별(Pulsar Star) 관측
<div>
    <h5>1. 데이터</h5>
    <p>• 독립변수 : 통합프로파일(Integrated Profile) 평균, 표준편차 등 8개 특성</p>
    <p>• 종속변수 : 펄사 별(Pulsar Star) 여부</p>
    <br>
    <h5>2. 기본 모델 성능 : 이진 판단</h5>
    <p>• Epoch : 10 , lr : 0.001</p>
    <p>• Train Loss : 1.014</p>
    <p>• Test<br>
        &nbsp;&nbsp;Acc = 0.976, Precision = 0.926, Recall = 0.789, F1 = 0.852</p>
    <br>
    <h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
    <div class="text-center indent_0"><a class="btn btn-outline-danger"
        href="{% url 'GPTeachersDay:eda_star' %}">
        EDA 별 : 바로가기
    </a></div>
</div>


<br><br><hr>


## AI 모델 3 : 철판 불량 검출
<div>
    <h5>1. 데이터</h5>
    <p>• 독립변수 : 불량 이미지 픽셀넓이, 휘도, 철판 두께 등 27개 특성</p>
    <p>• 종속변수 : Pastry, Z_Scratch, 오염 등 7가지 불량</p>
    <br>
    <h5>2. 기본 모델 성능 : 다중 분류</h5>
    <p>• Epoch : 10 , lr : 0.001</p>
    <p>• Train Loss : 16.039</p>
    <p>• Train Accuracy : 0.303</p>
    <p>• Test Accuracy : 0.412</p>
    <br>
    <h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
    <div class="text-center indent_0"><a class="btn btn-outline-success"
        href="{% url 'GPTeachersDay:eda_steel' %}">
        EDA 철판 : 바로가기
    </a></div>
</div>


<br><br><hr>


## 멤버 소개

<table style="width : 80%; margin : auto;">
    <tbody style="width : 100%; display : table;">
        <tr style="border-bottom : 3px solid gray; background-color : #88bb88;">
            <th style="width : 30%; text-align : center;">Name</th>
            <th style="width : 40%; text-align : center;">Role</th>
            <th style="width : 30%; text-align : center;">Link</th>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">문제성</td>
            <td style="width : 40%; text-align : start;">
                • Header <br> • EDA <br> • Modeling
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/GPTeachersDay" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">김동현</td>
            <td style="width : 40%; text-align : start;">
                • Modeling <br> • Custom Accuracy
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/GPTeachersDay" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">권순범</td>
            <td style="width : 40%; text-align : start;">
                • Modeling <br> • Optimizer
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/GPTeachersDay" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">이연준</td>
            <td style="width : 40%; text-align : start;">
                • EDA <br> • Engineering
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/leeyeonjun85" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
    </tbody>
</table>

<br><br><hr>

## git 관련 bash 명령어
```bash
# 브랜치 상태 확인
git status
# 브랜치 만들기
git branch yeonjun3
# 브랜치 삭제
git branch -d yeonjun3
# 브랜치 이동
git checkout yeonjun3
```


<br><br><hr>


## 링크
- <a href="http://leeyj85.shop/GPTeachersDay" target="_blank">WEB 프로젝트 페이지</a>
- <a href="https://codestates.notion.site/AIB-17-Team-Project-1-2023-05-15-2023-05-25-9454e090dcdf4cf891c71c0b4bd2ba5e" target="_blank">프로젝트 노션</a>
- <a href="https://www.notion.so/9891e517ff9a473491a1d4d2f3a87221?v=d776e70e97454284b0cc4c6988a77a51" target="_blank">팀 노션페이지</a>
- <a href="https://www.notion.so/1-1-23de33f86c034ca4836fb0d45bbad632" target="_blank">1일차 노션</a>
- <a href="https://www.notion.so/1-2-20fbb27c574f409a838f22aeeab6636d" target="_blank">2일차 노션</a>
- <a href="https://www.notion.so/1-3-8df24c40ff3146aaa7f1adf8fc1a1f3a" target="_blank">3일차 노션</a>
- <a href="https://www.notion.so/1-4-f0011339e35143f7a98daff17746856e" target="_blank">4일차 노션</a>
- <a href="https://www.notion.so/1-5-74bbb5f192324074ab4042312ba97c5c" target="_blank">5일차 노션</a>
- <a href="https://www.notion.so/1-5-74bbb5f192324074ab4042312ba97c5c" target="_blank">5일차 노션</a>
- <a href="https://www.notion.so/1-6-d71da6cae65446a8805f29ce147c5c37" target="_blank">6일차 노션</a>
- <a href="https://www.notion.so/1-7-70feca0a849544c09cf007c061084982" target="_blank">7일차 노션</a>
- <a href="https://www.notion.so/1-8-c0f4d48ebc0245c3b767df3d687acd08" target="_blank">8일차 노션</a>
- <a href="https://www.notion.so/1-9-09ea6055070d4ea59b0fd6369f5bae7b" target="_blank">9일차 노션</a>
- <a href="https://www.notion.so/1-10-1272b6eaf94d4bdf8eabf293bb1901ce" target="_blank">10일차 노션</a>