import pandas as pd

class Agent1:
    def __init__(self, binary_predictor, regression_predictor, labels=["label1", "label2"]):
        self.binary_predictor = binary_predictor
        self.regression_predictor = regression_predictor
        self.labels = labels

    def act(self, prev_data, capital, holding_coin):
        """
        prev_data: {coin: row(Series)} - 어제 데이터 (모델 입력용)
        capital: 현재 자본
        ------------------------
        return: action dict
            {
                "type": "BUY" | "SELL" | "HOLD",
                "coin": str or None,
                "price": float or None,
                "reason": str (optional)
            }
        """
        # -----------------------------
        # 1. 모델 추론
        # -----------------------------
        X_batch = []
        coins = []
        for c, row in prev_data.items():
            features = row.drop(self.labels, errors="ignore")
            X_batch.append(features)
            coins.append(c)
        X_batch = pd.DataFrame(X_batch)

        probs = self.binary_predictor.predict_proba(X_batch)[1].values
        preds_prob = {coin: prob for coin, prob in zip(coins, probs)}
        vals = self.regression_predictor.predict(X_batch)
        preds_val = {coin: (0.02 + (val / 100 - 0.02) * 0.5) for coin, val in zip(coins, vals)}
        preds_exp = {coin: prob * (0.02 + (val / 100 - 0.02) * 0.5) for coin, prob, val in zip(coins, probs, vals)}

        now_best_coin = None
        while True:
            best_coin = max(preds_exp, key=preds_exp.get)
            if preds_exp[best_coin] <= 0:  # 기준치 미달 → 매수 안 함
                break

            # 유동성 체크 (거래 금액이 자본 * 200 이상)
            if prev_data[best_coin]["volume"] * prev_data[best_coin]["close"] < capital * 100:
                preds_exp[best_coin] = -1.0
                continue
            if preds_prob[best_coin] < 0.65:  # 기준치 미달 → 매수 안 함
                preds_exp[best_coin] = -1.0
                continue
            now_best_coin = best_coin
            break

        order_lst = []

        # -----------------------------
        # 2. 보유 중일 경우: 매도 조건 확인
        # -----------------------------
        if holding_coin is not None:

            # 모델 신뢰도 낮으면 매도
            if preds_prob[holding_coin] < 0.65:
                order_lst.append({"type": "MARKET_SELL", "coin": holding_coin, "price": 0, "reason": "low_prob"})
                holding_coin = None
            else:
                order_lst.extend([
                    {"type": "LIMIT_SELL_MIN", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * 0.99, "reason": "stop_loss"},
                    {"type": "LIMIT_SELL_MAX", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * (1 + preds_val[holding_coin]), "reason": "take_profit"}
                ])
        
        # -----------------------------
        # 3. 보유 중이지 않다면: 매수 조건 확인
        # -----------------------------
        if holding_coin is None:
            if now_best_coin is not None:
                # 매수 실행
                order_lst.append({"type": "MARKET_BUY", "coin": now_best_coin, "price": 0, "reason": "model_signal"})
                order_lst.extend([
                    {"type": "LIMIT_SELL_MIN", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * 0.99, "reason": "stop_loss"},
                    {"type": "LIMIT_SELL_MAX", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * (1 + preds_val[now_best_coin]), "reason": "take_profit"}
                ])

        return order_lst