# claude-video-hyperframes

Short videos built with [HyperFrames](https://hyperframes.heygen.com). Each video is its own project folder.

```
true-me/                 TRUE ME page videos (Instagram / Facebook reels, 1080x1920)
  letting-go/            "Letting Go" - source (index.html), piano.mp3, renders/
learning-tests/          practice and test videos
  sleep-benefits/        8s landscape test: battery-recharge metaphor
```

## Preview or render a video

```
npm install                       # once, from the repo root
cd true-me/letting-go
npx hyperframes preview           # opens Studio at http://localhost:3002
npx hyperframes render -o renders/my-video.mp4 --fps 24 -q draft
```

## New TRUE ME video

```
cd true-me
npx hyperframes init my-new-video --example blank --resolution portrait
```
