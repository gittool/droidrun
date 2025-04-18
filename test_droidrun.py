#!/usr/bin/env python3
import asyncio
import os
import json
import logging
from typing import Dict, List, Any, Optional
from droidrun.agent.react_agent import ReActAgent
import google.generativeai as genai
from dotenv import load_dotenv

# ロガーを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("droidrun-test")

# .env ファイルから環境変数を読み込む
load_dotenv()

# APIキーを確認
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")

# Gemini APIを設定
genai.configure(api_key=api_key)

class GeminiReasoner:
    """Gemini APIを直接使ったLLMレゾナー"""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        vision: bool = False
    ):
        """Geminiレゾナーを初期化
        
        Args:
            model_name: 使用するGeminiモデル名
            temperature: 生成の温度パラメータ
            max_tokens: 生成する最大トークン数
            vision: 画像機能（スクリーンショット）が有効かどうか
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vision = vision
        
        # トークン使用量の追跡
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        
        # Geminiモデルの初期化
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                }
            )
            logger.info(f"Geminiモデル {self.model_name} を初期化しました")
        except Exception as e:
            logger.error(f"Geminiモデルの初期化エラー: {e}")
            raise
    
    def get_token_usage_stats(self) -> Dict[str, int]:
        """現在のトークン使用統計を取得
        
        Returns:
            トークン使用統計を含む辞書
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls
        }
    
    async def reason(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
        screenshot_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """LLMを使用して推論ステップを生成
        
        Args:
            goal: 自動化の目標
            history: 以前のステップのリスト（辞書として）
            available_tools: 利用可能なツール名の任意のリスト
            screenshot_data: 最新のスクリーンショットを含むバイトデータ（任意）
        
        Returns:
            次の推論ステップを含む辞書（思考、アクション、パラメータを含む）
        """
        # API呼び出し前の現在のトークン使用統計を表示
        logger.info(f"API呼び出し前のトークン使用: {self.get_token_usage_stats()}")
        
        # プロンプトを構築
        system_prompt = self._create_system_prompt(available_tools)
        user_prompt = self._create_user_prompt(goal, history)
        
        try:
            # スクリーンショットがある場合のマルチモーダル処理
            if screenshot_data and self.vision:
                import base64
                # 画像バイトをbase64に変換
                base64_image = base64.b64encode(screenshot_data).decode('utf-8')
                
                # マルチモーダルコンテンツを作成
                content = [
                    {"text": "Here's the current screenshot of the device. Please analyze it to help with the next action."},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }}
                ]
                
                # ユーザープロンプトを追加
                content.append({"text": user_prompt})
                
                # Gemini APIを呼び出し
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    content=content,
                    system_instruction=system_prompt,
                )
            else:
                # テキストのみの呼び出し
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    contents=[system_prompt, user_prompt],
                )
            
            # レスポンステキストを取得
            response_text = response.text
            
            # APIコール回数をカウント
            self.api_calls += 1
            
            # トークン使用量を推定（Google APIはOpenAIのように使用量を返さないため、簡易推定）
            estimated_prompt_tokens = len(system_prompt + user_prompt) // 4
            estimated_completion_tokens = len(response_text) // 4
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            
            # 統計を更新
            self.total_prompt_tokens += estimated_prompt_tokens
            self.total_completion_tokens += estimated_completion_tokens
            self.total_tokens += estimated_total_tokens
            
            # トークン使用情報を表示
            logger.info("===== トークン使用統計（推定） =====")
            logger.info(f"API呼び出し #{self.api_calls}")
            logger.info(f"今回: {estimated_prompt_tokens} プロンプト + {estimated_completion_tokens} 完了 = {estimated_total_tokens} トークン")
            logger.info(f"累計: {self.total_prompt_tokens} プロンプト + {self.total_completion_tokens} 完了 = {self.total_tokens} トークン")
            logger.info("=================================")
            
            # レスポンスを解析
            return self._parse_response(response_text)
            
        except Exception as e:
            logger.error(f"Gemini API呼び出しエラー: {e}")
            # フォールバックレスポンス
            return {
                "thought": f"LLM推論エラー: {e}",
                "action": "error",
                "parameters": {}
            }
    
    def _create_system_prompt(self, available_tools: Optional[List[str]] = None) -> str:
        """LLM用のシステムプロンプトを作成
        
        Args:
            available_tools: 利用可能なツール名の任意のリスト
        
        Returns:
            システムプロンプト文字列
        """
        # 基本システムプロンプト
        prompt = """
        あなたはAndroidフォン用のユーザーアシスタントです。指定された目標を達成するためにAndroidデバイスを制御することがタスクです。
        以下のガイドラインに従ってください：

        1. 現在の画面状態をUI状態から分析し、すべてのUI要素を取得する
        2. ステップバイステップで考えて行動を計画する
        3. 各ステップに最も適切なツールを選択する
        4. 以下のフィールドを含むJSON形式でレスポンスを返す：
        - thought: 現在の状態と次に何をすべきかについての詳細な推論
        - action: 実行するツールの名前（括弧なしでツール名を正確に使用）
        - parameters: ツールに渡すパラメータの辞書

        重要：actionフィールドを指定する際：
        - ツール名に括弧を追加しないでください
        - よくある間違い：
          ❌ "get_clickables()"
          ✅ "get_clickables"

        観察のための2つの重要なツールがあります：
        1. 現在の画面をより良く理解するために、画面に含まれるすべてのテキストを含むすべてのUI要素を取得できます。これを使用して現在のUIコンテキストを分析してください。
        2. アクションを起こしたい場合は、コンテキストを分析した後、次の対話ステップのためにクリック可能なすべての要素を取得できます。現在のUIコンテキストについて知っている場合にのみこのツールを使用してください。
        """
        
        # ビジョンが有効な場合、特定の指示を追加
        if self.vision:
            prompt += """
            take_screenshotツールを通じてスクリーンショットにアクセスできます。視覚的なコンテキストが必要な場合に使用してください。
            """
        else:
            prompt += """
            ビジョンは無効です。get_clickablesからのテキストベースのUI要素データのみに頼ってください。
            """
                
        # ツールのドキュメント（正確なパラメータ名付き）
        tool_docs = {
            "tap": "tap(index: int) - 指定されたインデックスの要素をデバイス上でタップする",
            
            "swipe": "swipe(start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) - (start_x,start_y)から(end_x,end_y)までduration_msミリ秒かけてスワイプする",
            
            "input_text": "input_text(text: str) - デバイスにテキストを入力する - これは入力フィールドがフォーカスされている場合にのみ機能します。テキストを挿入する前に編集フィールドがタップされていることを常に確認してください",
            
            "press_key": "press_key(keycode: int) - キーコードを使用してデバイス上でキーを押す",
            
            "start_app": "start_app(package: str, activity: str = '') - パッケージ名を使用してアプリを起動する（例：'com.android.settings'）",
            
            "list_packages": "list_packages(include_system_apps: bool = False) - デバイスにインストールされているパッケージを一覧表示し、詳細なパッケージ情報を返す",
            
            "get_clickables": "get_clickables() - デバイス画面からクリック可能なUI要素のみを取得します。対話型要素とそのプロパティを含む辞書を返します",

            "complete": "complete(result: str) - 重要：このツールは、目標に必要なすべての操作を実際に完了した後にのみ呼び出す必要があります。これ自体はアクションを実行せず、他のアクションを通じて目標を達成したことを示すだけです。結果パラメータとして達成内容の要約を含めてください。",
        }
        
        # ビジョンが有効な場合のみ、take_screenshotツールを追加
        if self.vision:
            tool_docs["take_screenshot"] = "take_screenshot() - 現在のUIをよりよく理解するためにスクリーンショットを撮ります。視覚的コンテキストが必要なときに使用してください。"
        
        # 利用可能なツール情報が提供されている場合に追加
        if available_tools:
            prompt += "\n\n利用可能なツールとそのパラメータ：\n"
            
            # 利用可能なツールのドキュメントのみを含める
            for tool in available_tools:
                if tool in tool_docs:
                    prompt += f"- {tool_docs[tool]}\n"
                else:
                    prompt += f"- {tool}（パラメータ不明）\n"
        
        return prompt
    
    def _create_user_prompt(
        self,
        goal: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """LLM用のユーザープロンプトを作成
        
        Args:
            goal: 自動化の目標
            history: 過去のステップのリスト
        
        Returns:
            ユーザープロンプト文字列
        """
        prompt = f"目標: {goal}\n\n"
        
        # 履歴がある場合は追加
        if history:
            # トークンのための予算から開始（非常に粗い近似）
            total_budget = 100000  # レスポンス用のスペースを残すための控えめな制限
            
            # 目標とその他の部分のトークンを推定
            goal_tokens = self._estimate_tokens(goal) * 2  # 繰り返しを考慮
            
            # 履歴用の残りの予算を計算
            history_budget = total_budget - goal_tokens
            
            # 最新の履歴から始めて、遡って処理
            truncated_history = []
            current_size = 0
            
            # 最新のものを先に処理するために履歴をコピーして反転
            reversed_history = list(reversed(history))
            
            for step in reversed_history:
                step_type = step.get("type", "").upper()
                content = step.get("content", "")
                step_text = f"{step_type}: {content}\n"
                step_tokens = self._estimate_tokens(step_text)
                
                # このステップが予算を超える場合、追加を停止
                if current_size + step_tokens > history_budget:
                    # 切り詰めに関する注記を追加
                    truncated_history.insert(0, "...（より古い履歴は切り詰められました）")
                    break
                
                # そうでなければ、このステップを追加して現在のサイズを更新
                truncated_history.insert(0, step_text)
                current_size += step_tokens
            
            # 切り詰められた履歴をプロンプトに追加
            prompt += "履歴:\n"
            for step_text in truncated_history:
                prompt += step_text
            prompt += "\n"
        
        prompt += "現在の状態に基づいて、次のアクションは何ですか？JSON形式でレスポンスを返してください。"
        
        # 最終的な健全性チェック - プロンプトがまだ大きすぎる場合、積極的に切り詰める
        if self._estimate_tokens(prompt) > 100000:
            logger.warning("通常の切り詰め後もプロンプトが大きすぎます。緊急切り詰めを適用します。")
            # 始め（目標）と終わり（指示）を保持するが、中央部分を切り詰める
            beginning = prompt[:2000]  # 目標を保持
            end = prompt[-1000:]       # 最終指示を保持
            prompt = beginning + "\n...（コンテンツはトークン制限に収まるように切り詰められました）...\n" + end
        
        return prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """文字列内のトークン数を推定
        
        これは英語テキストの場合、1トークンが約4文字であるという経験則に基づく
        非常に大雑把な近似です。
        
        Args:
            text: 入力テキスト
        
        Returns:
            推定トークン数
        """
        if not text:
            return 0
        return len(text) // 4 + 1  # 安全のために1を追加
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """LLMレスポンスを構造化形式に解析
        
        Args:
            response: LLMレスポンス文字列
        
        Returns:
            解析されたレスポンスを含む辞書
        """
        try:
            # JSONとして解析を試みる
            data = json.loads(response)
            
            # 必要なフィールドが存在することを確認
            if "thought" not in data:
                data["thought"] = "思考が提供されていません"
            if "action" not in data:
                data["action"] = "no_action"
            if "parameters" not in data:
                data["parameters"] = {}
                
            return data
        except json.JSONDecodeError:
            # 有効なJSONでない場合、正規表現を使ってフィールドを抽出
            import re
            thought_match = re.search(r'thought["\s:]+([^"]+)', response)
            action_match = re.search(r'action["\s:]+([^",\n]+)', response)
            params_match = re.search(r'parameters["\s:]+({.+})', response, re.DOTALL)
            
            thought = thought_match.group(1) if thought_match else "思考の解析に失敗しました"
            action = action_match.group(1) if action_match else "no_action"
            
            # パラメータの解析を試みる
            params = {}
            if params_match:
                try:
                    params_str = params_match.group(1)
                    # 有効なJSONのためにシングルクォートをダブルクォートに置き換え
                    params_str = params_str.replace("'", "\"")
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    logger.warning("パラメータJSONの解析に失敗しました")
            
            return {
                "thought": thought,
                "action": action,
                "parameters": params
            }

async def main():
    # Gemini APIを使用したレゾナーを作成
    gemini_reasoner = GeminiReasoner(
        model_name="gemini-2.5-flash-preview-04-17",
        temperature=0.2
    )
    
    # エージェントを作成して実行
    agent = ReActAgent(
        task="Open the Settings app and check the Android version(日本語で表示されます,日本語で会話しよう)",
        llm=gemini_reasoner  # GeminiReasonerインスタンスを使用
    )
    # chromeを開いて, cb-cloud.comのサイトを表示する(日本語で会話しよう)
    steps = await agent.run()
    print(f"実行が {len(steps)} ステップで完了しました")

if __name__ == "__main__":
    asyncio.run(main())
