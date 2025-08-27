import pandas as pd

class Agent1:
    def __init__(self, binary_predictor1, binary_predictor2, labels=["label1", "label2"]):
        self.binary_predictor1 = binary_predictor1
        self.binary_predictor2 = binary_predictor2
        self.labels = labels
        self.preds1_momentum = None
        self.preds2_momentum = None
        self.preds_momentum_sum = 0.0

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

        if self.preds1_momentum is None:
            self.preds1_momentum = {coin: 0.0 for coin in coins}
            self.preds2_momentum = {coin: 0.0 for coin in coins}

        probs1 = self.binary_predictor1.predict_proba(X_batch)[1].values
        preds_prob1 = {coin: prob for coin, prob in zip(coins, probs1)}
        probs2 = self.binary_predictor2.predict_proba(X_batch)[1].values
        preds_prob2 = {coin: prob1 * prob2 for coin, prob1, prob2 in zip(coins, probs1, probs2)}

        final_preds_prob1 = {prob1[0]: (prob1[1] + momentum[1]) / (self.preds_momentum_sum + 1.0) for prob1, momentum in zip(preds_prob1.items(), self.preds1_momentum.items())}
        final_preds_prob2 = {prob2[0]: (prob2[1] + momentum[1]) / (self.preds_momentum_sum + 1.0) for prob2, momentum in zip(preds_prob2.items(), self.preds1_momentum.items())}
        for coin in coins:
            self.preds1_momentum[coin] = (self.preds1_momentum[coin] + preds_prob1[coin]) * 0.92
            self.preds2_momentum[coin] = (self.preds2_momentum[coin] + preds_prob2[coin]) * 0.92
        self.preds_momentum_sum = (self.preds_momentum_sum + 1.0) * 0.92

        now_best_coin = None
        while True:
            best_coin = max(final_preds_prob2, key=final_preds_prob2.get)
            if final_preds_prob2[best_coin] < 0.65:  # 기준치 미달 → 매수 안 함
                break

            # 유동성 체크 (거래 금액이 자본 * 200 이상)
            if prev_data[best_coin]["volume"] * prev_data[best_coin]["close"] < capital * 100:
                final_preds_prob2[best_coin] = -1.0
                continue
            now_best_coin = best_coin
            break
    
        if now_best_coin is None:
            while True:
                best_coin = max(final_preds_prob1, key=final_preds_prob1.get)
                if final_preds_prob1[best_coin] < 0.65:  # 기준치 미달 → 매수 안 함
                    break

                # 유동성 체크 (거래 금액이 자본 * 200 이상)
                if prev_data[best_coin]["volume"] * prev_data[best_coin]["close"] < capital * 100:
                    final_preds_prob1[best_coin] = -1.0
                    continue
                now_best_coin = best_coin
                break
        order_lst = []

        # -----------------------------
        # 2. 보유 중일 경우: 매도 조건 확인
        # -----------------------------
        if holding_coin is not None:

            # 모델 신뢰도 낮으면 매도
            if final_preds_prob1[holding_coin] < 0.6 or (now_best_coin is not None and final_preds_prob2[holding_coin] < 0.6 and final_preds_prob2[now_best_coin] > 0.65):    # holding_coin != now_best_coin
                order_lst.append({"type": "MARKET_SELL", "coin": holding_coin, "price": 0, "reason": "low_prob"})
                holding_coin = None
            else:
                if final_preds_prob2[holding_coin] < 0.65:
                    order_lst.extend([
                        {"type": "LIMIT_SELL_MIN", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * 0.99, "reason": "stop_loss"},
                        {"type": "LIMIT_SELL_MAX", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * 1.02, "reason": "take_profit"}
                    ])
                else:
                    order_lst.extend([
                        {"type": "LIMIT_SELL_MIN", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * 0.99, "reason": "stop_loss"},
                        {"type": "LIMIT_SELL_MAX", "coin": holding_coin, "price": prev_data[holding_coin]["close"] * 1.04, "reason": "take_profit"}
                    ])
        
        # -----------------------------
        # 3. 보유 중이지 않다면: 매수 조건 확인
        # -----------------------------
        if holding_coin is None:
            if now_best_coin is not None:
                # 매수 실행
                order_lst.append({"type": "MARKET_BUY", "coin": now_best_coin, "price": 0, "reason": "model_signal"})
                if final_preds_prob2[now_best_coin] < 0.65:
                    order_lst.extend([
                        {"type": "LIMIT_SELL_MIN", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * 0.99, "reason": "stop_loss"},
                        {"type": "LIMIT_SELL_MAX", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * 1.02, "reason": "take_profit"}
                    ])
                else:
                    order_lst.extend([
                        {"type": "LIMIT_SELL_MIN", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * 0.99, "reason": "stop_loss"},
                        {"type": "LIMIT_SELL_MAX", "coin": now_best_coin, "price": prev_data[now_best_coin]["close"] * 1.04, "reason": "take_profit"}
                    ])

        return order_lst