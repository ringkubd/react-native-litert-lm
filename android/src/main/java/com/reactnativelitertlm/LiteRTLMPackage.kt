package com.reactnativelitertlm

import android.view.View
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ReactShadowNode
import com.facebook.react.uimanager.ViewManager

/**
 * React Native package for auto-linking LiteRTLMModule.
 *
 * This package is registered automatically by React Native autolinking
 * when `react-native-litert-lm` is added as a dependency.
 *
 * It also works with Expo Modules API through `expo-modules-core`.
 */
class LiteRTLMPackage : ReactPackage {

    override fun createNativeModules(
        reactContext: ReactApplicationContext,
    ): List<NativeModule> {
        return listOf(LiteRTLMModule(reactContext))
    }

    override fun createViewManagers(
        reactContext: ReactApplicationContext,
    ): List<ViewManager<out View, out ReactShadowNode<*>>> {
        return emptyList()
    }
}
