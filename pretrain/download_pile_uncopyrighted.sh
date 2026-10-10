#!/usr/bin/env bash
# 下载 monology/pile-uncopyrighted：30 个训练分片，以及验证集和测试集。
# 用法：bash pretrain/download_pile_uncopyrighted.sh /path/to/pile

set -euo pipefail

if [[ $# -eq 1 && ( $1 == -h || $1 == --help ) ]]; then
    echo "用法：bash $0 /path/to/pile"
    exit 0
fi
if [[ $# -ne 1 ]]; then
    echo "用法：bash $0 /path/to/pile" >&2
    exit 2
fi

for command_name in curl awk wc tr mktemp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "缺少命令：$command_name" >&2
        exit 1
    fi
done
if command -v sha256sum >/dev/null 2>&1; then
    hash_command=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    hash_command=shasum
else
    echo "缺少 SHA-256 校验命令：sha256sum 或 shasum" >&2
    exit 1
fi

max_attempts=${PILE_MAX_ATTEMPTS:-10}
if [[ ! $max_attempts =~ ^[1-9][0-9]*$ ]]; then
    echo "PILE_MAX_ATTEMPTS 必须是正整数" >&2
    exit 2
fi

base_url=${PILE_BASE_URL:-https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main}
base_url=${base_url%/}
mkdir -p -- "$1/train" "$1/val" "$1/test"
root=$(cd -- "$1" && pwd -P)
headers_file=$(mktemp "$root/.pile-headers.XXXXXX")
trap 'rm -f -- "$headers_file"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    echo "错误：$*" >&2
    exit 1
}

file_size() {
    wc -c < "$1" | tr -d '[:space:]'
}

file_sha256() {
    if [[ $hash_command == sha256sum ]]; then
        sha256sum -- "$1" | awk '{print $1}'
    else
        shasum -a 256 -- "$1" | awk '{print $1}'
    fi
}

retryable() {
    local curl_code=$1 http_code=$2
    case "$http_code" in
        429|5??) return 0 ;;
        4??) return 1 ;;
    esac
    case "$curl_code" in
        6|7|18|28|35|52|55|56) return 0 ;;
        *) return 1 ;;
    esac
}

wait_before_retry() {
    local attempt=$1 label=$2
    local delay=$((2 ** attempt))
    if (( delay > 30 )); then delay=30; fi
    echo "$label：第 $attempt 次失败，${delay} 秒后重试（最多 $max_attempts 次）。" >&2
    sleep "$delay"
}

# HEAD 跟随镜像跳转。最终响应的 Content-Length 是文件大小；镜像的
# X-Linked-ETag 是文件 SHA-256，可避免把错误内容当成完整文件。
read_metadata() {
    local url=$1 label=$2 attempt status curl_code metadata final_size linked_size
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        : > "$headers_file"
        if status=$(curl --location --head --silent --show-error \
            --connect-timeout 20 --max-time 90 \
            --dump-header "$headers_file" --output /dev/null \
            --write-out '%{http_code}' -- "$url"); then
            curl_code=0
        else
            curl_code=$?
        fi

        if (( curl_code == 0 )) && [[ $status == 200 ]]; then
            metadata=$(awk '
                { sub(/\r$/, "") }
                /^HTTP\// { final_size = "" }
                tolower($1) == "content-length:" { final_size = $2 }
                tolower($1) == "x-linked-size:" { linked_size = $2 }
                tolower($1) == "x-linked-etag:" {
                    linked_sha = tolower($2)
                    gsub(/"/, "", linked_sha)
                }
                END { print final_size "|" linked_size "|" linked_sha }
            ' "$headers_file")
            IFS='|' read -r final_size linked_size expected_sha <<< "$metadata"
            expected_size=${final_size:-$linked_size}
            [[ $expected_size =~ ^[1-9][0-9]*$ ]] || fail "$label：无法取得有效文件大小"
            [[ $expected_sha =~ ^[0-9a-f]{64}$ ]] || fail "$label：无法取得有效 SHA-256"
            if [[ -n $linked_size && $linked_size != "$expected_size" ]]; then
                fail "$label：镜像与实际存储地址报告的文件大小不一致"
            fi
            return 0
        fi

        if (( attempt == max_attempts )) || ! retryable "$curl_code" "$status"; then
            fail "$label：获取文件信息失败（curl=$curl_code，HTTP=$status）"
        fi
        wait_before_retry "$attempt" "$label 的文件信息"
    done
}

verify_file() {
    local path=$1 label=$2 actual_size actual_sha
    actual_size=$(file_size "$path")
    [[ $actual_size == "$expected_size" ]] || fail "$label：文件大小不符（$actual_size / $expected_size 字节）：$path"
    echo "$label：正在校验 SHA-256..."
    actual_sha=$(file_sha256 "$path")
    [[ $actual_sha == "$expected_sha" ]] || fail "$label：SHA-256 不符，请检查文件：$path"
}

download_file() {
    local remote_path=$1 relative_path=$2
    local destination="$root/$relative_path" url="$base_url/$remote_path"
    local label=$relative_path part meta saved_size saved_sha attempt current_size status curl_code
    part="$destination.part"
    meta="$part.meta"
    echo "[$label] 读取远端文件信息..."
    read_metadata "$url" "$label"

    if [[ -e $destination ]]; then
        [[ -f $destination ]] || fail "$label：目标路径不是普通文件：$destination"
        verify_file "$destination" "$label"
        rm -f -- "$meta"
        echo "$label：已完整下载，跳过。"
        return 0
    fi

    if [[ -e $part ]]; then
        [[ -f $part && -f $meta ]] || fail "$label：存在没有元数据的残留 .part 文件：$part"
        read -r saved_size saved_sha < "$meta" || fail "$label：无法读取断点信息：$meta"
        [[ $saved_size == "$expected_size" && $saved_sha == "$expected_sha" ]] ||
            fail "$label：远端文件已变化，不能安全续传：$part"
    else
        # 先保存远端版本信息，再开始写 .part；下次运行据此判断能否续传。
        printf '%s %s\n' "$expected_size" "$expected_sha" > "$meta"
    fi

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        current_size=0
        if [[ -f $part ]]; then current_size=$(file_size "$part"); fi
        (( current_size <= expected_size )) || fail "$label：.part 大于远端文件：$part"
        if (( current_size == expected_size )); then
            verify_file "$part" "$label"
            mv -- "$part" "$destination"
            rm -f -- "$meta"
            echo "$label：完成。"
            return 0
        fi

        echo "$label：第 $attempt/$max_attempts 次下载，从 $current_size 字节开始。"
        local -a resume_option=(--retry 0)
        if (( current_size > 0 )); then resume_option=(-C -); fi
        if status=$(curl --fail --location --progress-bar --show-error \
            --connect-timeout 20 --speed-limit 1024 --speed-time 180 \
            "${resume_option[@]}" --output "$part" --write-out '%{http_code}' -- "$url"); then
            curl_code=0
        else
            curl_code=$?
        fi

        # 有断点时必须收到 206；200 表示服务器忽略了 Range，不能继续拼接。
        if (( current_size > 0 && curl_code == 0 )) && [[ $status != 206 ]]; then
            fail "$label：服务器未确认断点续传（HTTP=$status）：$part"
        fi
        if (( curl_code == 0 )); then
            current_size=$(file_size "$part")
            if (( current_size == expected_size )); then
                verify_file "$part" "$label"
                mv -- "$part" "$destination"
                rm -f -- "$meta"
                echo "$label：完成。"
                return 0
            fi
            (( current_size < expected_size )) || fail "$label：下载文件大于预期：$part"
            echo "$label：传输提前结束（$current_size / $expected_size 字节）。" >&2
        elif ! retryable "$curl_code" "$status"; then
            fail "$label：下载失败且不适合自动重试（curl=$curl_code，HTTP=$status）：$part"
        fi

        if (( attempt == max_attempts )); then
            fail "$label：达到最大重试次数，已保留断点文件：$part"
        fi
        wait_before_retry "$attempt" "$label"
    done
}

download_file val.jsonl.zst val/val.jsonl.zst
download_file test.jsonl.zst test/test.jsonl.zst
for ((i=0; i<30; i++)); do
    printf -v shard '%02d' "$i"
    download_file "train/$shard.jsonl.zst" "train/$shard.jsonl.zst"
done
echo "全部下载完成：$root"
