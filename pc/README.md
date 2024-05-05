# PlayCanvas (JS) + k (Rust)

✅ vr hands

✅ working server on pc

✅ working server on orin

⬜️

⬜️

## Orin

### Setup

```
sudo apt-get install npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 16
nvm use 16
npm install -g vite@latest
```

init dev repo (not needed if cloning this repo)

```
npm install playcanvas vite --save-dev
```

### Usage

run server

```
npx vite
```

Run [ngrok](https://ngrok.com/download).

```
ngrok http 5137
```

## References

https://developer.playcanvas.com/user-manual/engine/standalone/
https://github.com/playcanvas/engine

